import warnings

import dolfin as df
import numpy as np
import scipy.optimize
from ffc.quadrature.deprecation import \
    QuadratureRepresentationDeprecationWarning

from fenics_concrete import experimental_setups
from fenics_concrete.helpers import LocalProjector, Parameters, set_q
from fenics_concrete.material_problems.material_problem import MaterialProblem

df.parameters["form_compiler"]["representation"] = "quadrature"
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

# MaterialProblem class for AM including path variable for layer-by-layer simulation
# all problems are implemented in an incremental way with load increments
# (in case of density the load increments are computed automatic, for displacement controlled experiments user controlled)
# mechanics_problem available:
#       "ConcreteThixElasticModel": linear elastic with age dependent Young'modulus (Thixotropy model)
#       "ConcreteViscoDevElasticModel": viscoelastic (Three Parameter Model: CMaxwell or CKelvin) with deviator assumption
#       "ConcreteViscoDevThixElasticModel": above model with age dependent parameters (Thixotropy)
class ConcreteAMMechanical(MaterialProblem):
    def __init__(
        self,
        experiment=None,
        parameters=None,
        mech_prob_string=None,
        pv_name="pv_output_concrete-thix",
    ):
        # generate "dummy" experiment when none is passed
        if experiment == None:
            experiment = experimental_setups.get_experiment("MinimalCube", parameters)

        # set used mechanical problem
        self.mech_prob_string = mech_prob_string

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        # setup initial material parameters general ones
        default_p = Parameters()
        # Material parameter for concrete model with structural build-up
        default_p["density"] = 2070  # in kg/m^3 density of fresh concrete
        default_p["u_bc"] = 0.1  # displacement on top

        # polynomial degree
        default_p["degree"] = 2  # default boundary setting

        # material model parameters
        default_p["nu"] = 0.3  # Poissons Ratio

        # create model and set specific material parameters
        if self.mech_prob_string.lower() == "concretethixelasticmodel":
            ### default parameters required for thix elastic model
            # Youngs modulus is changing over age (see E_fkt) following the bilinear approach Kruger et al 2019
            # (https://www.sciencedirect.com/science/article/pii/S0950061819317507) with two different rates
            # random values as default
            default_p["E_0"] = 15000  # Youngs Modulus at age=0 in Pa
            default_p["R_E"] = 15  # Reflocculation rate in Pa / s
            default_p["A_E"] = 30  # Structuration rate in Pa / s
            default_p["t_f"] = 300  # Reflocculation time in s
            default_p["age_0"] = 0  # start age of concrete s
            default_p["load_time"] = 1  # load applied in 1 s
            self.p = default_p + self.p

            self.mechanics_problem = ConcreteThixElasticModel(
                self.experiment.mesh, self.p, pv_name=self.pv_name
            )

        elif self.mech_prob_string.lower() == "concreteviscodevelasticmodel":
            ### default parameters required for visco elastic model
            default_p["visco_case"] = "CMaxwell"  # maxwell body with spring in parallel
            default_p["E_0"] = 40000  # Youngs Modulus Pa linear elastic
            default_p["E_1"] = 20000  # Youngs Modulus Pa visco element
            default_p["eta"] = 1000  # Damping coeff
            default_p["load_time"] = 1  # load applied in 1 s
            self.p = default_p + self.p

            self.mechanics_problem = ConcreteViscoDevElasticModel(
                self.experiment.mesh, self.p, pv_name=self.pv_name
            )

        elif self.mech_prob_string.lower() == "concreteviscodevthixelasticmodel":
            ### default parameters required for visco elastic model
            default_p["visco_case"] = "CKelvin"  # maxwell body with spring in parallel
            default_p["E_0"] = 40000  # Youngs Modulus Pa linear elastic
            default_p["E_1"] = 20000  # Youngs Modulus Pa visco element
            default_p["eta"] = 1000  # Damping coeff
            default_p["R_i"] = [0, 0, 0]
            default_p["A_i"] = [0, 0, 0]
            default_p["t_f"] = [0, 0, 0]
            default_p["age_0"] = 0.0  # start age of concrete [s]
            default_p["load_time"] = 1  # load applied in 1 s
            self.p = default_p + self.p

            self.mechanics_problem = ConcreteViscoDevThixElasticModel(
                self.experiment.mesh, self.p, pv_name=self.pv_name
            )

        else:
            raise ValueError("given mechanics_problem not implemented")

        self.V = self.mechanics_problem.V  # for reaction force sensor
        self.residual = None  # initialize

        # setting bcs
        bcs = self.experiment.create_displ_bcs(self.mechanics_problem.V)

        self.mechanics_problem.set_bcs(bcs)

        # setting up the solver
        self.mechanics_solver = df.NewtonSolver()
        self.mechanics_solver.parameters["absolute_tolerance"] = 1e-7
        self.mechanics_solver.parameters["relative_tolerance"] = 1e-7

    def set_initial_path(self, path):
        self.mechanics_problem.set_initial_path(path)

    def solve(self, t=1.0):

        print("solve for", t)
        # CHANGED FOR INCREMENTAL SET UP from u to du!!!
        self.mechanics_solver.solve(
            self.mechanics_problem, self.mechanics_problem.du.vector()
        )

        # save fields to global problem for sensor output
        self.displacement = self.mechanics_problem.u

        self.stress = self.mechanics_problem.q_sig
        self.strain = self.mechanics_problem.q_eps
        # general interface if stress/strain are in voigt or full tensor format is specified in mechanics_problem!!
        self.visu_space_stress = self.mechanics_problem.visu_space_sig
        self.visu_space_strain = self.mechanics_problem.visu_space_eps

        # get sensor data
        self.residual = self.mechanics_problem.R  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

        # update age & path before next step!
        self.mechanics_problem.update_values()

    def pv_plot(self, t=0):
        # calls paraview output for both problems
        self.mechanics_problem.pv_plot(t=t)

    def set_timestep(self, dt):
        self.mechanics_problem.set_timestep(dt)

    def get_E_fkt(self):
        return np.vectorize(self.mechanics_problem.E_fkt)

    def set_inital_path(self, path):
        self.mechanics_problem.set_initial_path(path)


class ConcreteThixElasticModel(df.NonlinearProblem):
    # linear elasticity law with time depenendent stiffness parameter (Youngs modulus) modelling the thixotropy
    # tensor format!!
    # incremental formulated u= u_old + du solve for du with given load increment (using function q_fd)

    def __init__(self, mesh, p, pv_name="mechanics_output", **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError("Material law not implemented for 1D")
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        # todo: I do not like the "meshless" setup right now
        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + ".xdmf")
            self.pv_file.parameters["flush_output"] = True
            self.pv_file.parameters["functions_share_mesh"] = True
            # function space for single value per element, required for plot of quadrature space values

            #
            if self.p.degree == 1:
                self.visu_space = df.FunctionSpace(mesh, "DG", 0)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)

            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)

            # interface to problem for sensor output: # here tensor format is used for e_eps/q_sig
            self.visu_space_eps = self.visu_space_T
            self.visu_space_sig = self.visu_space_T

            metadata = {
                "quadrature_degree": self.p.degree,
                "quadrature_scheme": "default",
            }
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, "P", self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(
                q, cell, degree=self.p.degree, quad_scheme="default"
            )
            quadrature_vector_element = df.TensorElement(
                q, cell, degree=self.p.degree, quad_scheme="default"
            )
            # quadrature_vector_element01 = df.VectorElement(q, cell, degree=self.p.degree, dim=self.stress_vector_dim,
            #                                             quad_scheme="default")
            q_V = df.FunctionSpace(mesh, quadrature_element)
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element)  # full tensor

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density")  # active or nonactive
            self.q_E = df.Function(q_V, name="youngs modulus")
            self.q_fd = df.Function(q_V, name="load factor")  # for density
            self.q_eps = df.Function(q_VT, name="strain")
            self.q_sig = df.Function(q_VT, name="stress")
            self.q_dsig = df.Function(q_VT, name="delta stress")
            self.q_sig_old = df.Function(q_VT, name="old stress")

            # for incremental formulation
            self.u_old = df.Function(self.V, name="old displacement")
            self.u = df.Function(self.V, name="displacement")

            # Define variational problem
            self.du = df.Function(self.V)  # delta displacement
            v = df.TestFunction(self.V)

            # Volume force
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # define sigma from(u,t) in evalute material or here global E change ? (see damage example Thomas) -> then tangent by hand!
            # # Elasticity parameters without multiplication with E
            # self.sigma_ufl = self.q_E * self.x_sigma(self.u) # delta sigma

            # multiplication with activated elements / current Young's modulus
            R_ufl = self.q_E * df.inner(self.x_sigma(self.du), self.eps(v)) * dxm
            # add volumetric force, aka gravity (in this case) plus load factor
            R_ufl += -self.q_fd * df.inner(f, v) * dxm

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.du)
            # quadrature part
            self.dR = dR_ufl

            # stress and strain projection methods here as full tensors
            self.project_delta_sigma = LocalProjector(
                self.q_E * self.x_sigma(self.du), q_VT, dxm
            )
            self.project_strain = LocalProjector(self.eps(self.u), q_VT, dxm)

            self.assembler = None  # set as default, to check if bc have been added???

    def x_sigma(self, v):

        x_mu = 1.0 / (2.0 * (1.0 + self.p.nu))
        x_lambda = 1.0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
        if self.p.dim == 2 and self.p.stress_case == "plane_stress":
            # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
            x_lambda = 2 * x_mu * x_lambda / (x_lambda + 2 * x_mu)

        return 2.0 * x_mu * df.sym(df.grad(v)) + x_lambda * df.tr(
            df.sym(df.grad(v))
        ) * df.Identity(len(v))

    def eps(self, v):
        return df.sym(df.grad(v))

    def E_fkt(self, pd, path_time, parameters):
        # age dependent Young's modulus function here bilinear approach based on Kruger et al.

        if pd > 0:  # element active, compute current Young's modulus
            age = parameters["age_0"] + path_time  # age concrete
            if age < parameters["t_f"]:
                E = parameters["E_0"] + parameters["R_E"] * age
            elif age >= parameters["t_f"]:
                E = (
                    parameters["E_0"]
                    + parameters["R_E"] * parameters["t_f"]
                    + parameters["A_E"] * (age - parameters["t_f"])
                )
        else:
            E = df.DOLFIN_EPS  # non-active
            # E = 0.001 * parameters['E_0']  # Emin?? TODO: how to define Emin?

        return E

    def pd_fkt(self, path_time):
        # pseudo density: decide if layer is there (active) or not (age < 0 nonactive!)
        # decision based on current path_time value
        l_active = 0  # non-active
        if path_time >= 0 - df.DOLFIN_EPS:
            l_active = 1.0  # active
        return l_active

    def fd_fkt(self, pd, path_time, parameters):
        # load increment function: load linearly applied in "load_time" time interval
        fd = 0
        if pd > 0:  # element active compute current loading factor for density
            if path_time < parameters["load_time"]:
                fd = self.dt / parameters["load_time"]

        return fd

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        pd_list = pd_fkt_vectorized(
            path_list
        )  # current pseudo density 1 if path_time >=0 else 0
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus #TODO: maybe at n-1/2 instead
        param_E = {}
        param_E["t_f"] = self.p.t_f
        param_E["E_0"] = self.p.E_0
        param_E["R_E"] = self.p.R_E
        param_E["A_E"] = self.p.A_E
        param_E["age_0"] = self.p.age_0
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, param_E)
        # print('E',E_list.max(),E_list.min())

        # compute loading factors for density load
        param_fd = {}
        param_fd["load_time"] = self.p.load_time
        fd_list_vectorized = np.vectorize(self.fd_fkt)
        fd_list = fd_list_vectorized(pd_list, path_list, param_fd)
        # print("fd", fd_list.max(), fd_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)
        set_q(self.q_fd, fd_list)

        # displacement update for stress and strain computation (for visualization)
        # for total strain computation
        self.u.vector()[:] = self.u_old.vector()[:] + self.du.vector()[:]
        # get current total strains full tensor (split in old and delta not required)
        self.project_strain(self.q_eps)
        self.project_delta_sigma(self.q_dsig)  # get current stress delta full tensor
        self.q_sig.vector()[:] = self.q_sig_old.vector()[:] + self.q_dsig.vector()[:]

    def update_values(self):
        # no history field currently
        path_list = self.q_path.vector().get_local()
        path_list += self.dt * np.ones_like(path_list)

        set_q(self.q_path, path_list)

        # update old displacement state
        self.u_old.vector()[:] = np.copy(self.u.vector()[:])
        self.q_sig_old.vector()[:] = np.copy(self.q_sig.vector()[:])

    def set_timestep(self, dt):
        self.dt = dt

    def set_initial_path(self, path_time):
        self.q_path.interpolate(path_time)  # default = zero, given as expression

    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = df.SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        # if self.dt <= 0:
        #    raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)

    def pv_plot(self, t=0):
        # paraview export

        # displacement plot
        u_plot = df.project(self.u, self.V)
        u_plot.rename("Displacement", "displacemenet vector")
        self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        sigma_plot = df.project(
            self.q_sig,
            self.visu_space_T,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        eps_plot = df.project(
            self.q_eps,
            self.visu_space_T,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        # print('sigma plot', sigma_plot.vector()[:].max())
        E_plot = df.project(
            self.q_E,
            self.visu_space,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        pd_plot = df.project(
            self.q_pd,
            self.visu_space,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )

        E_plot.rename("Young's Modulus", "Young's modulus value")
        sigma_plot.rename("Stress", "stress components")
        pd_plot.rename("pseudo density", "pseudo density")
        eps_plot.rename("Strain", "strain components")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(eps_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)


class ConcreteViscoDevElasticModel(df.NonlinearProblem):
    # viscoelastic material law derived from 1D Three Parameter Model
    # two options: param['visco_case']=='Cmaxwell' -> Maxwell chain with n=1! == linear standard solid model (Maxwell in parallel with spring)
    #                                                ---------spring(E_0)-------
    #                                                |                          |
    #                                                --damper(eta)--spring(E_1)--
    #              param['visco_case']=='Ckelvin' --> Kelvin chain with n=1! == Kelvin plus spring (in Reihe)
    #                                                   ------spring(E_1)------
    #                                   ---spring(E_0)--|                     |
    #                                                   ------damper(eta)------
    # with deviatoric assumptions for 3D generalization:
    # Deviatoric assumption: vol part of visco strain == 0 damper just working on deviatoric part!
    # in tensor format!!
    # time integration: BACKWARD EULER
    # incremental u= uold + du -> solve for du with given load increment dF

    def __init__(self, mesh, p, pv_name="mechanics_output", **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError("Material law not implemented for 1D")
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + ".xdmf")
            self.pv_file.parameters["flush_output"] = True
            self.pv_file.parameters["functions_share_mesh"] = True
            # function space for single value per element, required for plot of quadrature space values

            #
            if self.p.degree == 1:
                self.visu_space = df.FunctionSpace(mesh, "DG", 0)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)
                # visu space for sigma and eps in voigt notation
                self.visu_space_V = df.VectorFunctionSpace(
                    mesh, "DG", 0, dim=self.stress_vector_dim
                )
            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)
                # visu space for sigma and eps in voigt notation
                self.visu_space_V = df.VectorFunctionSpace(
                    mesh, "P", 1, dim=self.stress_vector_dim
                )

            # interface to problem for sensor output! here VOIGT format is used for e_eps/q_sig
            self.visu_space_eps = self.visu_space_V
            self.visu_space_sig = self.visu_space_V

            metadata = {
                "quadrature_degree": self.p.degree,
                "quadrature_scheme": "default",
            }
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, "P", self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(
                q, cell, degree=self.p.degree, quad_scheme="default"
            )
            quadrature_vector_element = df.TensorElement(
                q, cell, degree=self.p.degree, quad_scheme="default"
            )
            quadrature_vector_element01 = df.VectorElement(
                q,
                cell,
                degree=self.p.degree,
                dim=self.stress_vector_dim,
                quad_scheme="default",
            )
            q_V = df.FunctionSpace(mesh, quadrature_element)
            # full tensor
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element)
            # voigt notation
            q_VTV = df.FunctionSpace(mesh, quadrature_vector_element01)

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density")  # active or nonactive
            self.q_E = df.Function(q_V, name="youngs modulus")  # not age-depend
            self.q_fd = df.Function(q_V, name="load factor")  # for density
            self.q_epsv = df.Function(q_VT, name="visco strain")  # full tensor
            self.q_sig1_ten = df.Function(q_VT, name="tensor strain")  # full tensor
            # for visualization issues
            self.q_eps = df.Function(q_VTV, name="total strain")  # voigt notation
            self.q_sig = df.Function(q_VTV, name="total stress")  # voigt notation

            # for incremental formulation
            self.u_old = df.Function(self.V, name="old displacement")
            self.u = df.Function(self.V, name="displacement")

            # Define variational problem
            self.du = df.Function(self.V)  # current delta displacement
            v = df.TestFunction(self.V)

            # Volume force
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # multiplication with activated elements
            # part with eps
            R_ufl = self.q_E * df.inner(self.sigma(self.du), self.eps(v)) * dxm
            # visco part only where active
            R_ufl += -self.q_pd * df.inner(self.sigma_2(), self.eps(v)) * dxm
            # add volumetric force increments
            R_ufl += -self.q_fd * df.inner(f, v) * dxm

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.du)
            # quadrature part
            self.dR = dR_ufl

            # stress and strain projection methods working on full u (u_old + du)
            self.project_sigma = LocalProjector(
                self.sigma_voigt(self.sigma(self.u) - self.sigma_2()), q_VTV, dxm
            )
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VTV, dxm)
            # stress component for visco strain computation
            self.project_sig1_ten = LocalProjector(self.sigma_1(self.u), q_VT, dxm)

            self.assembler = None  # set as default, to check if bc have been added???

    def sigma(self, v):  # total stress without visco part
        mu_E0 = self.p.E_0 / (2.0 * (1.0 + self.p.nu))
        lmb_E0 = self.p.E_0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

        if self.p.dim == 2 and self.p.stress_case == "plane_stress":
            # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
            lmb_E0 = 2 * mu_E0 * lmb_E0 / (lmb_E0 + 2 * mu_E0)

        if self.p.visco_case.lower() == "cmaxwell":
            # stress stiffness zero + stress stiffness one
            sig = (
                2.0 * mu_E0 * self.eps(v)
                + lmb_E0 * df.tr(self.eps(v)) * df.Identity(self.p.dim)
                + self.sigma_1(v)
            )
        elif self.p.visco_case.lower() == "ckelvin":
            # stress stiffness zero
            sig = 2.0 * mu_E0 * self.eps(v) + lmb_E0 * df.tr(self.eps(v)) * df.Identity(
                self.p.dim
            )
        else:
            sig = None
            raise ValueError("case not defined")

        return sig

    def sigma_1(self, v):  # stress stiffness one
        if self.p.visco_case.lower() == "cmaxwell":
            mu_E1 = self.p.E_1 / (2.0 * (1.0 + self.p.nu))
            lmb_E1 = (
                self.p.E_1 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
            )
            if self.p.dim == 2 and self.p.stress_case == "plane_stress":
                lmb_E1 = 2 * mu_E1 * lmb_E1 / (lmb_E1 + 2 * mu_E1)
            sig1 = 2.0 * mu_E1 * self.eps(v) + lmb_E1 * df.tr(
                self.eps(v)
            ) * df.Identity(self.p.dim)
        elif self.p.visco_case.lower() == "ckelvin":
            sig1 = self.sigma(v)
        else:
            sig = None
            raise ValueError("case not defined")

        return sig1

    def sigma_2(self):  # related to epsv
        if self.p.visco_case.lower() == "cmaxwell":
            mu_E1 = self.p.E_1 / (2.0 * (1.0 + self.p.nu))
            sig2 = 2 * mu_E1 * self.q_epsv
        elif self.p.visco_case.lower() == "ckelvin":
            mu_E0 = self.p.E_0 / (2.0 * (1.0 + self.p.nu))
            sig2 = 2 * mu_E0 * self.q_epsv
        else:
            sig = None
            raise ValueError("case not defined")
        return sig2

    def eps(self, v):
        return df.sym(df.grad(v))

    def eps_voigt(self, e):
        eT = self.eps(e)
        # 2D option
        if eT.ufl_shape == (2, 2):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], 2 * eT[0, 1]))
        # 3D option
        elif eT.ufl_shape == (3, 3):
            strain_vector = df.as_vector(
                (eT[0, 0], eT[1, 1], eT[2, 2], 2 * eT[0, 1], 2 * eT[1, 2], 2 * eT[0, 2])
            )

        return strain_vector

    def sigma_voigt(self, s):
        # 2D option
        if s.ufl_shape == (2, 2):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[0, 1]))
        # 3D option
        elif s.ufl_shape == (3, 3):
            stress_vector = df.as_vector(
                (s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2])
            )
        else:
            raise ("Problem with stress tensor shape for voigt notation")

        return stress_vector

    def E_fkt(self, pd, path_time, parameters):

        if pd > 0:  # element active, compute current Young's modulus
            E = 1.0  # no thixotropy evaluation yet!!!
        else:
            E = df.DOLFIN_EPS  # non-active
            # E = 0.001 * parameters['E_0']  # Emin??

        return E

    def pd_fkt(self, path_time):
        # pseudo denisty: decide if layer is active or not (age < 0 nonactive!)
        # decision based on current path_time value
        l_active = 0  # non-active
        if path_time >= 0 - df.DOLFIN_EPS:
            l_active = 1.0  # active
        return l_active

    def fd_fkt(self, pd, path_time, parameters):
        # load increment function
        fd = 0
        if pd > 0:  # element active compute current loading factor for density
            if path_time < parameters["load_time"]:
                fd = self.dt / parameters["load_time"]

        return fd

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        # current pseudo density 1 if path_time >=0 else 0
        pd_list = pd_fkt_vectorized(path_list)
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus
        parameters = {}
        parameters["E_0"] = self.p.E_0
        #
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, parameters)
        # print('E',E_list.max(),E_list.min())

        # compute loading factors for density load
        param_fd = {}
        param_fd["load_time"] = self.p.load_time
        fd_list_vectorized = np.vectorize(self.fd_fkt)
        fd_list = fd_list_vectorized(pd_list, path_list, param_fd)
        # print("fd", fd_list.max(), fd_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)
        set_q(self.q_fd, fd_list)

        # displacement update for stress and strain computation
        self.u.vector()[:] = self.u_old.vector()[:] + self.du.vector()[:]
        # self.u.vector()[:] = self.du.vector()[:] # if not incremental

        # get current strains and stresses
        self.project_sig1_ten(self.q_sig1_ten)  # get stress component

        # old visco strains (= deviatoric part)
        epsv_list = self.q_epsv.vector().get_local()
        sig1_list = self.q_sig1_ten.vector().get_local()

        # compute visco strain from old epsv
        self.new_epsv = np.zeros_like(epsv_list)

        if self.p.visco_case.lower() == "cmaxwell":
            mu_E1 = 0.5 * self.p.E_1 / (1.0 + self.p.nu)
            factor = 1 + self.dt * 2.0 * mu_E1 / self.p.eta
            self.new_epsv = (
                1.0 / factor * (epsv_list + self.dt / self.p.eta * sig1_list)
            )
        elif self.p.visco_case.lower() == "ckelvin":
            mu_E1 = 0.5 * self.p.E_1 / (1.0 + self.p.nu)
            mu_E0 = 0.5 * self.p.E_0 / (1.0 + self.p.nu)
            factor = (
                1
                + self.dt * 2.0 * mu_E0 / self.p.eta
                + self.dt * 2.0 * mu_E1 / self.p.eta
            )
            self.new_epsv = (
                1.0 / factor * (epsv_list + self.dt / self.p.eta * sig1_list)
            )
        else:
            raise ValueError("visco case not defined")

        # for sensors and visualization
        self.project_strain(self.q_eps)  # get current strains voigt notation
        self.project_sigma(self.q_sig)  # get current stress voigt notation

    def update_values(self):
        # update process time and path variable
        path_list = self.q_path.vector().get_local()
        path_list += self.dt * np.ones_like(path_list)

        set_q(self.q_path, path_list)

        # update visco strain
        set_q(self.q_epsv, self.new_epsv)

        # update old displacement state
        self.u_old.vector()[:] = np.copy(self.u.vector()[:])

    def set_timestep(self, dt):
        self.dt = dt

    def set_initial_path(self, path_time):
        self.q_path.interpolate(path_time)  # default = zero, given as expression

    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = df.SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        # if self.dt <= 0:
        #    raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)

    def pv_plot(self, t=0):
        # paraview export

        # displacement plot
        u_plot = df.project(self.u, self.V)
        u_plot.rename("Displacement", "displacemenet vector")
        self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        sigma_plot = df.project(
            self.q_sig,
            self.visu_space_V,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        eps_plot = df.project(
            self.q_eps,
            self.visu_space_V,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        E_plot = df.project(
            self.q_E,
            self.visu_space,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        pd_plot = df.project(
            self.q_pd,
            self.visu_space,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )

        E_plot.rename("Young's Modulus", "Young's modulus value")
        pd_plot.rename("pseudo density", "pseudo density")
        sigma_plot.rename("Stress", "stress components voigt")
        eps_plot.rename("strain", "strain components voigt")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(eps_plot, t, encoding=df.XDMFFile.Encoding.ASCII)


class ConcreteViscoDevThixElasticModel(df.NonlinearProblem):
    # viscoelastic material law derived from 1D Three Parameter Model
    # two options: param['visco_case']=='Cmaxwell' -> Maxwell chain with n=1! == linear standard solid model (Maxwell in parallel with spring)
    #                                                ---------spring(E_0)-------
    #                                                |                          |
    #                                                --damper(eta)--spring(E_1)--
    #              param['visco_case']=='Ckelvin' --> Kelvin chain with n=1! == Kelvin plus spring (in Reihe)
    #                                                   ------spring(E_1)------
    #                                   ---spring(E_0)--|                     |
    #                                                   ------damper(eta)------
    # with deviatoric assumptions for 3D generalization:
    # Deviatoric assumption: vol part of visco strain == 0 damper just working on deviatoric part!
    # in tensor format!!
    # time integration: BACKWARD EULER
    # incremental u= uold + du -> solve for du with given load increment dF
    # with the option of time dependent parameters E_0(t), E_1(t), eta(t)

    def __init__(self, mesh, p, pv_name="mechanics_output", **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError("Material law not implemented for 1D")
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + ".xdmf")
            self.pv_file.parameters["flush_output"] = True
            self.pv_file.parameters["functions_share_mesh"] = True
            # function space for single value per element, required for plot of quadrature space values

            #
            if self.p.degree == 1:
                self.visu_space = df.FunctionSpace(mesh, "DG", 0)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)
                # visu space for sigma and eps in voigt notation
                self.visu_space_V = df.VectorFunctionSpace(
                    mesh, "DG", 0, dim=self.stress_vector_dim
                )
            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)
                # visu space for sigma and eps in voigt notation
                self.visu_space_V = df.VectorFunctionSpace(
                    mesh, "P", 1, dim=self.stress_vector_dim
                )

            # interface to problem for sensor output: here VOIGT format
            self.visu_space_eps = self.visu_space_V
            self.visu_space_sig = self.visu_space_V

            metadata = {
                "quadrature_degree": self.p.degree,
                "quadrature_scheme": "default",
            }
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, "P", self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(
                q, cell, degree=self.p.degree, quad_scheme="default"
            )
            quadrature_vector_element = df.TensorElement(
                q, cell, degree=self.p.degree, quad_scheme="default"
            )
            quadrature_vector_element01 = df.VectorElement(
                q,
                cell,
                degree=self.p.degree,
                dim=self.stress_vector_dim,
                quad_scheme="default",
            )
            q_V = df.FunctionSpace(mesh, quadrature_element)
            # full tensor
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element)
            # voigt notation
            q_VTV = df.FunctionSpace(mesh, quadrature_vector_element01)

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density")  # active or nonactive
            self.q_E0 = df.Function(q_V, name="elastic modulus")  # age dependent
            self.q_E1 = df.Function(q_V, name="visco modulus")  # age dependent
            self.q_eta = df.Function(q_V, name="damper modulus")  # age dependent
            self.q_fd = df.Function(q_V, name="load factor")  # for density
            self.q_epsv = df.Function(q_VT, name="visco strain")  # full tensor
            self.q_sig1_ten = df.Function(q_VT, name="tensor strain")  # full tensor
            # for visualization issues
            self.q_eps = df.Function(q_VTV, name="total strain")  # voigt notation
            self.q_sig = df.Function(q_VTV, name="total stress")  # voigt notation

            # for incremental formulation
            self.u_old = df.Function(self.V, name="old displacement")
            self.u = df.Function(self.V, name="displacement")

            # Define variational problem
            self.du = df.Function(self.V)  # current delta displacement
            v = df.TestFunction(self.V)

            # Volume force
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # multiplication with activated elements
            R_ufl = df.inner(self.sigma(self.du), self.eps(v)) * dxm  # part with eps
            # visco part only where active
            R_ufl += -self.q_pd * df.inner(self.sigma_2(), self.eps(v)) * dxm
            # add volumetric force increment
            R_ufl += -self.q_fd * df.inner(f, v) * dxm

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.du)
            # quadrature part
            self.dR = dR_ufl

            # stress and strain projection methods
            self.project_sigma = LocalProjector(
                self.sigma_voigt(self.sigma(self.u) - self.sigma_2()), q_VTV, dxm
            )
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VTV, dxm)
            self.project_sig1_ten = LocalProjector(
                self.sigma_1(self.u), q_VT, dxm
            )  # stress component for visco strain computation

            self.assembler = None  # set as default, to check if bc have been added???

    def sigma(self, v):  # total stress without visco part
        mu_E0 = self.q_E0 / (2.0 * (1.0 + self.p.nu))
        lmb_E0 = self.q_E0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

        if self.p.dim == 2 and self.p.stress_case == "plane_stress":
            # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
            lmb_E0 = 2 * mu_E0 * lmb_E0 / (lmb_E0 + 2 * mu_E0)
        if self.p.visco_case.lower() == "cmaxwell":
            # stress stiffness zero + stress stiffness one
            sig = (
                2.0 * mu_E0 * self.eps(v)
                + lmb_E0 * df.tr(self.eps(v)) * df.Identity(self.p.dim)
                + self.sigma_1(v)
            )
        elif self.p.visco_case.lower() == "ckelvin":
            sig = 2.0 * mu_E0 * self.eps(v) + lmb_E0 * df.tr(self.eps(v)) * df.Identity(
                self.p.dim
            )  # stress stiffness zero
        else:
            sig = None
            raise ValueError("case not defined")

        return sig

    def sigma_1(self, v):  # stress for visco strain computation
        if self.p.visco_case.lower() == "cmaxwell":
            mu_E1 = self.q_E1 / (2.0 * (1.0 + self.p.nu))
            lmb_E1 = (
                self.q_E1 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
            )
            if self.p.dim == 2 and self.p.stress_case == "plane_stress":
                lmb_E1 = 2 * mu_E1 * lmb_E1 / (lmb_E1 + 2 * mu_E1)
            sig1 = 2.0 * mu_E1 * self.eps(v) + lmb_E1 * df.tr(
                self.eps(v)
            ) * df.Identity(self.p.dim)
        elif self.p.visco_case.lower() == "ckelvin":
            sig1 = self.sigma(v)
        else:
            sig = None
            raise ValueError("case not defined")

        return sig1

    def sigma_2(self):  # damper stress related to epsv
        if self.p.visco_case.lower() == "cmaxwell":
            mu_E1 = self.q_E1 / (2.0 * (1.0 + self.p.nu))
            sig2 = 2 * mu_E1 * self.q_epsv
        elif self.p.visco_case.lower() == "ckelvin":
            mu_E0 = self.q_E0 / (2.0 * (1.0 + self.p.nu))
            sig2 = 2 * mu_E0 * self.q_epsv
        else:
            sig = None
            raise ValueError("case not defined")
        return sig2

    def eps(self, v):
        return df.sym(df.grad(v))

    def eps_voigt(self, e):
        eT = self.eps(e)
        # 2D option
        if eT.ufl_shape == (2, 2):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], 2 * eT[0, 1]))
        # 3D option
        elif eT.ufl_shape == (3, 3):
            strain_vector = df.as_vector(
                (eT[0, 0], eT[1, 1], eT[2, 2], 2 * eT[0, 1], 2 * eT[1, 2], 2 * eT[0, 2])
            )

        return strain_vector

    def sigma_voigt(self, s):
        # 2D option
        if s.ufl_shape == (2, 2):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[0, 1]))
        # 3D option
        elif s.ufl_shape == (3, 3):
            stress_vector = df.as_vector(
                (s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2])
            )
        else:
            raise ("Problem with stress tensor shape for voigt notation")

        return stress_vector

    def E_fkt(self, pd, path_time, param):
        # bilinear fkt for material parameter P
        # fkt defined by param-dict: {'P0','R','A','t_f'}
        # P0 start value; R first slope; A second slope; t_f time at which slope changes
        if pd > 0:  # element active, compute current Young's modulus
            age = self.p.age_0 + path_time  # age concrete
            if age < param["t_f"]:
                P = param["P0"] + param["R"] * age
            elif age >= param["t_f"]:
                P = (
                    param["P0"]
                    + param["R"] * param["t_f"]
                    + param["A"] * (age - param["t_f"])
                )
        else:
            P = df.DOLFIN_EPS  # non-active

        return P

    def pd_fkt(self, path_time):
        # pseudo denisty: decide if layer is active or not (age < 0 nonactive!)
        # decision based on current path_time value
        l_active = 0  # non-active
        if path_time >= 0 - df.DOLFIN_EPS:
            l_active = 1.0  # active
        return l_active

    def fd_fkt(self, pd, path_time, parameters):
        # load increment function
        fd = 0
        if pd > 0:  # element active compute current loading factor for density
            if path_time < parameters["load_time"]:
                fd = self.dt / parameters["load_time"]

        return fd

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        pd_list = pd_fkt_vectorized(
            path_list
        )  # current pseudo density 1 if path_time >=0 else 0
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E0_list = E_fkt_vectorized(
            pd_list,
            path_list,
            {
                "P0": self.p.E_0,
                "R": self.p.R_i["E_0"],
                "A": self.p.A_i["E_0"],
                "t_f": self.p.t_f["E_0"],
            },
        )
        E1_list = E_fkt_vectorized(
            pd_list,
            path_list,
            {
                "P0": self.p.E_1,
                "R": self.p.R_i["E_1"],
                "A": self.p.A_i["E_1"],
                "t_f": self.p.t_f["E_1"],
            },
        )
        eta_list = E_fkt_vectorized(
            pd_list,
            path_list,
            {
                "P0": self.p.eta,
                "R": self.p.R_i["eta"],
                "A": self.p.A_i["eta"],
                "t_f": self.p.t_f["eta"],
            },
        )
        # print("E0", E0_list.max(), E0_list.min(), len(E0_list))

        # compute loading factors for density load
        param_fd = {}
        param_fd["load_time"] = self.p.load_time
        fd_list_vectorized = np.vectorize(self.fd_fkt)
        fd_list = fd_list_vectorized(pd_list, path_list, param_fd)
        # print("fd", fd_list.max(), fd_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E0, E0_list)
        set_q(self.q_E1, E1_list)
        set_q(self.q_eta, eta_list)

        set_q(self.q_pd, pd_list)
        set_q(self.q_fd, fd_list)

        # displacement update for stress and strain computation
        self.u.vector()[:] = self.u_old.vector()[:] + self.du.vector()[:]
        # self.u.vector()[:] = self.du.vector()[:] # if not incremental

        # get current strains and stresses
        self.project_sig1_ten(self.q_sig1_ten)  # get stress component

        # old visco strains (= deviatoric part)
        epsv_list = self.q_epsv.vector().get_local()
        sig1_list = self.q_sig1_ten.vector().get_local()

        # compute visco strain from old one epsv
        self.new_epsv = np.zeros_like(epsv_list)

        if self.p.visco_case.lower() == "cmaxwell":
            mu_E1 = (
                0.5 * E1_list / (1.0 + self.p.nu)
            )  # list of E1 at each quadrature point
            factor = 1 + self.dt * 2.0 * mu_E1 / eta_list  # at each quadrature point

            # repeat material parameters to size of epsv and compute epsv
            self.new_epsv = (
                1.0
                / np.repeat(factor, self.p.dim**2)
                * (
                    epsv_list
                    + self.dt / np.repeat(eta_list, self.p.dim**2) * sig1_list
                )
            )  # reshaped material parameters per eps entry!!
        elif self.p.visco_case.lower() == "ckelvin":
            mu_E1 = 0.5 * E1_list / (1.0 + self.p.nu)
            mu_E0 = 0.5 * E0_list / (1.0 + self.p.nu)
            factor = (
                1 + self.dt * 2.0 * mu_E0 / eta_list + self.dt * 2.0 * mu_E1 / eta_list
            )

            # repeat material parameters to size of epsv and compute epsv
            self.new_epsv = (
                1.0
                / np.repeat(factor, self.p.dim**2)
                * (
                    epsv_list
                    + self.dt / np.repeat(eta_list, self.p.dim**2) * sig1_list
                )
            )
        else:
            raise ValueError("visco case not defined")

        # for sensors and visualization
        self.project_strain(self.q_eps)  # get current strains voigt notation
        self.project_sigma(self.q_sig)  # get current stress voigt notation

    def update_values(self):
        # update process time and path variable
        path_list = self.q_path.vector().get_local()
        path_list += self.dt * np.ones_like(path_list)

        set_q(self.q_path, path_list)

        # update visco strain
        set_q(self.q_epsv, self.new_epsv)

        # update old displacement state
        self.u_old.vector()[:] = np.copy(self.u.vector()[:])

    def set_timestep(self, dt):
        self.dt = dt

    def set_initial_path(self, path_time):
        self.q_path.interpolate(path_time)  # default = zero, given as expression

    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = df.SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        # if self.dt <= 0:
        #    raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)

    def pv_plot(self, t=0):
        # paraview export

        # displacement plot
        u_plot = df.project(self.u, self.V)
        u_plot.rename("Displacement", "displacemenet vector")
        self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        sigma_plot = df.project(
            self.q_sig,
            self.visu_space_V,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        eps_plot = df.project(
            self.q_eps,
            self.visu_space_V,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        E_plot = df.project(
            self.q_E0,
            self.visu_space,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )
        pd_plot = df.project(
            self.q_pd,
            self.visu_space,
            form_compiler_parameters={"quadrature_degree": self.p.degree},
        )

        E_plot.rename("elastic Young's Modulus", "elastic Young's modulus value")
        pd_plot.rename("pseudo density", "pseudo density")
        sigma_plot.rename("Stress", "stress components voigt")
        eps_plot.rename("strain", "strain components voigt")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(eps_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
