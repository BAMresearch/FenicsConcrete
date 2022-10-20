import dolfin as df
import numpy as np
import scipy.optimize


from fenics_concrete.material_problems.material_problem import MaterialProblem

from fenics_concrete.helpers import Parameters
from fenics_concrete.helpers import set_q
from fenics_concrete.helpers import LocalProjector
from fenics_concrete import experimental_setups


import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

df.parameters["form_compiler"]["representation"] = "quadrature"
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

# MaterialProblem class for AM including path variable for layer-by-layer simulation
# mechanics_problem choosable with "mech_prob" in the moment: "ConcreteThixElasticModel" or "ConcreteViscoElasticModel"
class ConcreteAMMechanical(MaterialProblem):
    def __init__(self, experiment=None, parameters=None, mech_prob_string=None, pv_name='pv_output_concrete-thix'):
        # generate "dummy" experiment when none is passed
        if experiment == None:
            experiment = experimental_setups.get_experiment('MinimalCube', parameters)

        # set used mechanical problem
        self.mech_prob_string = mech_prob_string

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        # setup initial material parameters general ones
        default_p = Parameters()
        # Material parameter for concrete model with structural build-up
        default_p['density'] = 2070  # in kg/m^3 density of fresh concrete see Wolfs et al 2018
        default_p['u_bc'] = 0.1 # displacement on top

        # polynomial degree
        default_p['degree'] = 2  # default boundary setting

        # material model parameters
        default_p['nu'] = 0.3  # Poissons Ratio see Wolfs et al 2018

        # create model and set specific material parameters
        if self.mech_prob_string.lower() =='concretethixelasticmodel':
            ### default parameters required for thix elastic model
            # Youngs modulus is changing over age (see E_fkt) following the bilinear approach Kruger et al 2019
            # (https://www.sciencedirect.com/science/article/pii/S0950061819317507) with two different rates
            default_p['E_0'] = 15000  # Youngs Modulus at age=0 in Pa # here random values!!
            default_p['R_E'] = 15  # Reflocculation rate of E modulus from age=0 to age=t_f in Pa / s
            default_p['A_E'] = 30  # Structuration rate of E modulus from age >= t_f in Pa / s
            default_p['t_f'] = 300  # Reflocculation time (switch between reflocculation rate and structuration rate) in s
            default_p['age_0'] = 0  # start age of concrete [s]
            self.p = default_p + self.p

            self.mechanics_problem = ConcreteThixElasticModel(self.experiment.mesh, self.p, pv_name=self.pv_name)

        elif self.mech_prob_string.lower() =='concreteviscoelasticmodel':
            ### default parameters required for visco elastic model
            default_p['E_0'] = 40000  # Youngs Modulus Pa linear elastic
            default_p['E_1'] = 20000  # Youngs Modulus Pa visco element
            default_p['eta'] = 1000  # Damping coeff
            self.p = default_p + self.p

            self.mechanics_problem = ConcreteViscoElasticModel(self.experiment.mesh, self.p, pv_name=self.pv_name)
        elif self.mech_prob_string.lower() =='concreteviscodevelasticmodel':
            ### default parameters required for visco elastic model
            default_p['visco_case'] = 'CMaxwell' # maxwell body with spring in parallel
            default_p['E_0'] = 40000  # Youngs Modulus Pa linear elastic
            default_p['E_1'] = 20000  # Youngs Modulus Pa visco element
            default_p['eta'] = 1000  # Damping coeff
            self.p = default_p + self.p

            self.mechanics_problem = ConcreteViscoDevElasticModel(self.experiment.mesh, self.p, pv_name=self.pv_name)
        else:
            raise ValueError('given mechanics_problem not implemented')


        self.V = self.mechanics_problem.V # for reaction force sensor
        self.residual = None # initialize

        # setting bcs
        bcs = self.experiment.create_displ_bcs(self.mechanics_problem.V) # fixed boundary bottom

        self.mechanics_problem.set_bcs(bcs)

        # setting up the solver
        self.mechanics_solver = df.NewtonSolver()
        self.mechanics_solver.parameters['absolute_tolerance'] = 1e-7
        self.mechanics_solver.parameters['relative_tolerance'] = 1e-7

    def set_initial_path(self, path):
        self.mechanics_problem.set_initial_path(path)

    def solve(self, t=1.0):

        print('solve for',t)
        self.mechanics_solver.solve(self.mechanics_problem, self.mechanics_problem.u.vector())

        # save fields to global problem for sensor output
        self.displacement = self.mechanics_problem.u

        try: # if possible tensor functions from ufl formulations
            self.stress = self.mechanics_problem.sigma_ufl
            self.strain = self.mechanics_problem.eps(self.mechanics_problem.u)
            self.visu_space_T = self.mechanics_problem.visu_space_T
        except: # use quadrature fct defined in law and project in sensor onto vector fct space
            self.stress = self.mechanics_problem.q_sig
            self.strain = self.mechanics_problem.q_eps
            self.visu_space_V = self.mechanics_problem.visu_space_V

        # further ?? stress/strain ...

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


class ConcreteThixElasticModel(df.NonlinearProblem):
    # linear elasticity law with time depenendent stiffness parameter (Youngs modulus) modelling the thixotropy
    # tensor format

    def __init__(self, mesh, p, pv_name='mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError('Material law not implemented for 1D')
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        # todo: I do not like the "meshless" setup right now
        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + '.xdmf')
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

            metadata = {"quadrature_degree": self.p.degree, "quadrature_scheme": "default"}
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, 'P', self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(q, cell, degree=self.p.degree, quad_scheme="default")
            # quadrature_vector_element = df.VectorElement(q, cell, degree=self.p.degree, dim=self.stress_vector_dim,
            #                                              quad_scheme="default")
            q_V = df.FunctionSpace(mesh, quadrature_element)
            # q_VT = df.FunctionSpace(mesh, quadrature_vector_element)

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")  # negative values where not active yet

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density") # active or nonactive
            self.q_E = df.Function(q_V, name="youngs modulus")
            # self.q_sigma = df.Function(q_VT, name="stress")
            # self.q_eps = df.Function(q_VT, name="Strain")

            # Define variational problem
            self.u = df.Function(self.V)  # displacement
            v = df.TestFunction(self.V)

            # Volume force
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # define sigma from(u,t) in evalute material or here global E change ? (see damage example Thomas) -> then tangent by hand!
            # # Elasticity parameters without multiplication with E
            # self.x_mu = 1.0 / (2.0 * (1.0 + self.p.nu))
            # self.x_lambda = 1.0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
            self.sigma_ufl = self.q_E * self.x_sigma(self.u)

            # multiplication with activated elements / current Young's modulus
            R_ufl = self.q_E * df.inner(self.x_sigma(self.u), self.eps(v)) * dxm
            R_ufl += - self.q_pd * df.inner(f, v) * dxm  # add volumetric force, aka gravity (in this case)

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.u)
            # quadrature part
            self.dR = dR_ufl

            # self.project_sigma = LocalProjector(self.sigma_voigt(self.sigma_ufl), q_VT, dxm)
            # self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VT, dxm)

            self.assembler = None  # set as default, to check if bc have been added???

    def x_sigma(self, v):

        x_mu = 1.0 / (2.0 * (1.0 + self.p.nu))
        x_lambda = 1.0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
        if self.p.dim ==2 and self.p.stress_case == 'plane_stress':
            x_lambda = 2 * x_mu * x_lambda / (x_lambda + 2 * x_mu) # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html

        return 2.0 * x_mu * df.sym(df.grad(v)) + x_lambda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))

    def eps(self,v):
        return df.sym(df.grad(v))

    def sigma_voigt(self, s):
        # 1D option
        print('s.ufl_shape', s.ufl_shape)
        if s.ufl_shape == (1, 1):
            stress_vector = None
        # 2D option
        elif s.ufl_shape == (2, 2):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[0, 1]))
        # 3D option
        elif s.ufl_shape == (3, 3):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]))
        else:
            raise ('Problem with stress tensor shape for voigt notation')

        return stress_vector

    def eps_voigt(self, e):
        eT = self.eps(e)
        # 1D option
        if eT.ufl_shape == (1, 1):
            strain_vector = None
        # 2D option
        elif eT.ufl_shape == (2, 2):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], 2 * eT[0, 1]))
        # 3D option
        elif eT.ufl_shape == (3, 3):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], eT[2,2], 2 * eT[0, 1], 2*eT[1,2], 2*eT[0,2]))

        return strain_vector

    def E_fkt(self, pd, path_time, parameters):

        if pd > 0: # element active, compute current Young's modulus
            age = parameters['age_0'] + path_time # age concrete
            if age < parameters['t_f']:
                E = parameters['E_0'] + parameters['R_E'] * age
            elif age >= parameters['t_f']:
                E = parameters['E_0'] + parameters['R_E'] * parameters['t_f'] + parameters['A_E'] * (age-parameters['t_f'])
        else:
            E = df.DOLFIN_EPS # non-active
            # E = 0.001 * parameters['E_0']  # Emin?? TODO: how to define Emin?

        return E

    def pd_fkt(self, path_time):
        # pseudo denisty: decide if layer is active or not (age < 0 nonactive!)
        # decision based on current path_time value
        l_active = 0 # non-active
        if path_time >= 0-df.DOLFIN_EPS:
            l_active = 1.0 # active
        return l_active

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        pd_list = pd_fkt_vectorized(path_list) # current pseudo density 1 if path_time >=0 else 0
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus
        parameters = {}
        parameters['t_f'] = self.p.t_f
        parameters['E_0'] = self.p.E_0
        parameters['R_E'] = self.p.R_E
        parameters['A_E'] = self.p.A_E
        parameters['age_0'] = self.p.age_0
        #
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, parameters)
        # print('E',E_list.max(),E_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)

    def update_values(self):
        # no history field currently
        path_list = self.q_path.vector().get_local()
        path_list += self.dt * np.ones_like(path_list)

        set_q(self.q_path, path_list)

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

        sigma_plot = df.project(self.sigma_ufl, self.visu_space_T,
                                form_compiler_parameters={'quadrature_degree': self.p.degree})
        # print('sigma plot', sigma_plot.vector()[:].max())
        E_plot = df.project(self.q_E, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})
        pd_plot = df.project(self.q_pd, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})

        E_plot.rename("Young's Modulus", "Young's modulus value")
        sigma_plot.rename("Stress", "stress components")
        pd_plot.rename("pseudo density", "pseudo density")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)


class ConcreteViscoElasticModel(df.NonlinearProblem):
    # viscoelastic material law derived from 1D linear standard solid model (Maxwell body in parallel with spring)
    # with 3D with assumptions: 3D generalization where each element (2xsprings/damper) 2 moduli where the Poisson ratio is the same for all elements
    #                           see https://comet-fenics.readthedocs.io/en/latest/demo/viscoelasticity/linear_viscoelasticity.html
    # in VOIGT notation!! regarding Aratz first implementation
    # time integration: BACKWARD EULER

    def __init__(self, mesh, p, pv_name='mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError('Material law not implemented for 1D')
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + '.xdmf')
            self.pv_file.parameters["flush_output"] = True
            self.pv_file.parameters["functions_share_mesh"] = True
            # function space for single value per element, required for plot of quadrature space values

            #
            if self.p.degree == 1:
                self.visu_space = df.FunctionSpace(mesh, "DG", 0)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)
                self.visu_space_V = df.VectorFunctionSpace(mesh, 'DG', 0,
                                                           dim=self.stress_vector_dim)  # visu space for sigma and eps in voigt notation
            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)
                self.visu_space_V = df.VectorFunctionSpace(mesh, 'P', 1,
                                                           dim=self.stress_vector_dim)  # visu space for sigma and eps in voigt notation

            metadata = {"quadrature_degree": self.p.degree, "quadrature_scheme": "default"}
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, 'P', self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(q, cell, degree=self.p.degree, quad_scheme="default")
            quadrature_vector_element = df.VectorElement(q, cell, degree=self.p.degree, dim=self.stress_vector_dim,
                                                         quad_scheme="default")
            q_V = df.FunctionSpace(mesh, quadrature_element)
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element)

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")  # negative values where not active yet

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density")  # active or nonactive
            self.q_E = df.Function(q_V, name="youngs modulus")
            self.q_epsv = df.Function(q_VT, name='visco strain')
            self.q_eps = df.Function(q_VT, name='total strain')
            self.q_sig = df.Function(q_VT, name='total stress')

            # Define variational problem
            self.u = df.Function(self.V)  # full displacement
            v = df.TestFunction(self.V)

            # Volume force ??? correct?
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # multiplication with activated elements
            R_ufl = self.q_E * df.inner(self.sigma_1(self.u), self.eps_voigt(v)) * dxm # part with eps
            R_ufl += - self.q_pd * df.inner(self.sigma_2(), self.eps_voigt(v)) * dxm  # visco part
            R_ufl += - self.q_pd * df.inner(f, v) * dxm  # add volumetric force, aka gravity (in this case)

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.u)
            # quadrature part
            self.dR = dR_ufl

            # self.sigma_ufl = self.sigma_1(self.u)-self.sigma_2() # C_1 *eps + C_2 * (eps-epsv) # not possible because of internal strain variable
            # stress and strain projection methods
            self.project_sigma = LocalProjector(self.sigma_1(self.u)-self.sigma_2(), q_VT, dxm)
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VT, dxm)

            self.assembler = None  # set as default, to check if bc have been added???

    # Aratz version
    def sigma_1(self, v):  # related to eps
        return df.Constant(self.p.E_0) * self.dotC() * self.eps_voigt(v) + df.Constant(self.p.E_1) * self.dotC() * self.eps_voigt(v)

    def sigma_2(self):  # related to epsv
        return df.Constant(self.p.E_1) * self.dotC() * self.q_epsv

    def dotC(self):  # unit (E=1) linear elasticity matrix (Voigt notation)
        # nu: Poisson ratio
        nu = self.p.nu
        C = None
        if self.p.dim == 2:
            if self.p.stress_case == 'plane_stress':
                C = df.as_matrix([[1./(1.-nu**2), nu/(1.-nu**2), 0.],
                                  [nu/(1.-nu**2), 1./(1.-nu**2), 0.],
                                  [0., 0., (1-nu)/(2.*(1-nu**2))] ])
            else: # plane strain
                lmb_1 = 1.0 * nu / ((1. - 2. * nu) * (1. + nu))
                mu_1 = 0.5 * 1.0 / (1. + nu)
                C = df.as_matrix([[2. * mu_1 + lmb_1, lmb_1, 0],
                               [lmb_1, 2. * mu_1 + lmb_1, 0],
                               [0, 0, mu_1]])
        elif self.p.dim == 3:
            print('dim==3')
            lmb_1 = 1.0 * nu / ((1. - 2. * nu) * (1. + nu))
            mu_1 = 0.5 * 1.0 / (1. + nu)
            C = df.as_matrix([[2. * mu_1 + lmb_1, lmb_1, lmb_1, 0., 0., 0.],
                              [lmb_1, 2. * mu_1 + lmb_1, lmb_1, 0., 0., 0.],
                              [lmb_1, lmb_1, 2. * mu_1 + lmb_1, 0., 0., 0.],
                              [0., 0., 0., mu_1, 0., 0.],
                              [0., 0., 0., 0., mu_1, 0.],
                              [0., 0., 0., 0., 0., mu_1] ])

        return C

    def eps(self,v):
        return df.sym(df.grad(v))

    def eps_voigt(self, e):
        eT = self.eps(e)
        # 2D option
        if eT.ufl_shape == (2, 2):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], 2 * eT[0, 1]))
        # 3D option
        elif eT.ufl_shape == (3, 3):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], eT[2,2], 2 * eT[0, 1], 2*eT[1,2], 2*eT[0,2]))

        return strain_vector

    def E_fkt(self, pd, path_time, parameters):

        if pd > 0: # element active, compute current Young's modulus
            E = 1.0 # no thixotropy evaluation yet!!!
        else:
            E = df.DOLFIN_EPS # non-active
            # E = 0.001 * parameters['E_0']  # Emin??

        return E

    def pd_fkt(self, path_time):
        # pseudo denisty: decide if layer is active or not (age < 0 nonactive!)
        # decision based on current path_time value
        l_active = 0 # non-active
        if path_time >= 0-df.DOLFIN_EPS:
            l_active = 1.0 # active
        return l_active

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        pd_list = pd_fkt_vectorized(path_list) # current pseudo density 1 if path_time >=0 else 0
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus
        parameters = {}
        parameters['E_0'] = self.p.E_0
        #
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, parameters)
        # print('E',E_list.max(),E_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)

        # get current strains and stresses
        self.project_strain(self.q_eps)  # get current strains
        self.project_sigma(self.q_sig)  # get current stress
        eps_list = self.q_eps.vector().get_local()
        epsv_list = self.q_epsv.vector().get_local()  # old visco strains

        # compute visco strain from old one epsv point vice because of matrix multiplication! here with UFL
        # VERY SLOW BETTER tensor format with UFL or mcode interface
        self.new_epsv = np.zeros_like(epsv_list)
        num_gp = int(len(eps_list)/self.stress_vector_dim)
        C = self.dotC()  # get C matrix
        II = df.as_matrix(np.eye(self.stress_vector_dim))
        for i in range(num_gp):
            a = i*self.stress_vector_dim
            b = (i+1)*self.stress_vector_dim

            factor = II + self.dt * self.p.E_1 / self.p.eta * C
            factor_inv = df.inv(factor)
            self.new_epsv[a:b] = df.dot(factor_inv, df.as_vector(np.array(epsv_list[a:b]))) + \
                                 self.dt * self.p.E_1 / self.p.eta *df.dot(factor_inv, df.dot(C,df.as_vector(np.array(eps_list[a:b]))))

    def update_values(self):
        # update process time and path variable
        path_list = self.q_path.vector().get_local()
        path_list += self.dt * np.ones_like(path_list)

        set_q(self.q_path, path_list)

        # update visco strain
        set_q(self.q_epsv, self.new_epsv)

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

        # sigma_plot = df.project(self.sigma_ufl, self.visu_space_T,
        #                         form_compiler_parameters={'quadrature_degree': self.p.degree})
        # print('sigma plot', sigma_plot.vector()[:].max())
        sigma_plot = df.project(self.q_sig, self.visu_space_V, form_compiler_parameters={'quadrature_degree': self.p.degree})
        eps_plot = df.project(self.q_eps, self.visu_space_V,
                               form_compiler_parameters={'quadrature_degree': self.p.degree})
        E_plot = df.project(self.q_E, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})
        pd_plot = df.project(self.q_pd, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})


        E_plot.rename("Young's Modulus", "Young's modulus value")
        pd_plot.rename("pseudo density", "pseudo density")
        sigma_plot.rename("Stress", "stress components voigt")
        eps_plot.rename("strain", "strain components voigt")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(eps_plot, t, encoding=df.XDMFFile.Encoding.ASCII)


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

    def __init__(self, mesh, p, pv_name='mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError('Material law not implemented for 1D')
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + '.xdmf')
            self.pv_file.parameters["flush_output"] = True
            self.pv_file.parameters["functions_share_mesh"] = True
            # function space for single value per element, required for plot of quadrature space values

            #
            if self.p.degree == 1:
                self.visu_space = df.FunctionSpace(mesh, "DG", 0)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)
                self.visu_space_V = df.VectorFunctionSpace(mesh, 'DG', 0,
                                                           dim=self.stress_vector_dim)  # visu space for sigma and eps in voigt notation
            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)
                self.visu_space_V = df.VectorFunctionSpace(mesh, 'P', 1,
                                                           dim=self.stress_vector_dim)  # visu space for sigma and eps in voigt notation

            metadata = {"quadrature_degree": self.p.degree, "quadrature_scheme": "default"}
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, 'P', self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(q, cell, degree=self.p.degree, quad_scheme="default")
            quadrature_vector_element = df.TensorElement(q, cell, degree=self.p.degree, quad_scheme="default")
            quadrature_vector_element01 = df.VectorElement(q, cell, degree=self.p.degree, dim=self.stress_vector_dim,
                                                         quad_scheme="default")
            q_V = df.FunctionSpace(mesh, quadrature_element)
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element) # full tensor
            q_VTV = df.FunctionSpace(mesh, quadrature_vector_element01) # voigt notation

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")  # negative values where not active yet

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density")  # active or nonactive
            self.q_E = df.Function(q_V, name="youngs modulus")
            self.q_epsv = df.Function(q_VT, name='visco strain') # full tensor
            self.q_sig1_ten = df.Function(q_VT, name='tensor strain') # full tensor
            self.q_eps = df.Function(q_VTV, name='total strain') # voigt notation
            self.q_sig = df.Function(q_VTV, name='total stress') # voigt notation

            # Define variational problem
            self.u = df.Function(self.V)  # full displacement
            v = df.TestFunction(self.V)

            # Volume force ??? correct?
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # multiplication with activated elements
            R_ufl = self.q_E * df.inner(self.sigma(self.u), self.eps(v)) * dxm  # part with eps
            R_ufl += - self.q_pd * df.inner(self.sigma_2(), self.eps(v)) * dxm  # visco part
            R_ufl += - self.q_pd * df.inner(f, v) * dxm  # add volumetric force, aka gravity (in this case)

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.u)
            # quadrature part
            self.dR = dR_ufl

            # stress and strain projection methods
            self.project_sigma = LocalProjector(self.sigma_voigt(self.sigma(self.u) - self.sigma_2()), q_VTV, dxm)
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VTV, dxm)
            self.project_sig1_ten = LocalProjector(self.sigma_1(self.u), q_VT, dxm) # stress component for visco strain computation

            self.assembler = None  # set as default, to check if bc have been added???

    def sigma(self, v): #total stress without visco part
        mu_E0 = self.p.E_0 / (2.0 * (1.0 + self.p.nu))
        lmb_E0 = self.p.E_0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

        if self.p.dim == 2 and self.p.stress_case == 'plane_stress':
            lmb_E0 = 2 * mu_E0 * lmb_E0 / (
                        lmb_E0 + 2 * mu_E0)  # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html

        if self.p.visco_case.lower() == 'cmaxwell':
            sig = 2.0 * mu_E0 * self.eps(v) + lmb_E0 * df.tr(self.eps(v)) * df.Identity(self.p.dim) + self.sigma_1(v) # stress stiffness zero + stress stiffness one
        elif self.p.visco_case.lower() == 'ckelvin':
            sig = 2.0 * mu_E0 * self.eps(v) + lmb_E0 * df.tr(self.eps(v)) * df.Identity(self.p.dim) # stress stiffness zero
        else:
            sig = None
            raise ValueError('case not defined')


        return sig

    def sigma_1(self, v): #stress stiffness one
        if self.p.visco_case.lower() == 'cmaxwell':
            mu_E1 = self.p.E_1 / (2.0 * (1.0 + self.p.nu))
            lmb_E1 = self.p.E_1 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
            if self.p.dim ==2 and self.p.stress_case == 'plane_stress':
                lmb_E1 = 2 * mu_E1 * lmb_E1 / (lmb_E1 + 2 * mu_E1)
            sig1 = 2.0 * mu_E1 * self.eps(v) + lmb_E1 * df.tr(self.eps(v)) * df.Identity(self.p.dim)
        elif self.p.visco_case.lower() == 'ckelvin':
            sig1 = self.sigma(v)
        else:
            sig = None
            raise ValueError('case not defined')

        return sig1

    def sigma_2(self):  # related to epsv
        if self.p.visco_case.lower() == 'cmaxwell':
            mu_E1 = self.p.E_1 / (2.0 * (1. + self.p.nu))
            sig2 = 2 * mu_E1 * self.q_epsv
        elif self.p.visco_case.lower() == 'ckelvin':
            mu_E0 = self.p.E_0 / (2.0 * (1. + self.p.nu))
            sig2 = 2 * mu_E0 * self.q_epsv
        else:
            sig = None
            raise ValueError('case not defined')
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
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], eT[2, 2], 2 * eT[0, 1], 2 * eT[1, 2], 2 * eT[0, 2]))

        return strain_vector

    def sigma_voigt(self, s):
        # 2D option
        if s.ufl_shape == (2, 2):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[0, 1]))
        # 3D option
        elif s.ufl_shape == (3, 3):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]))
        else:
            raise ('Problem with stress tensor shape for voigt notation')

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

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        pd_list = pd_fkt_vectorized(path_list)  # current pseudo density 1 if path_time >=0 else 0
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus
        parameters = {}
        parameters['E_0'] = self.p.E_0
        #
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, parameters)
        # print('E',E_list.max(),E_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)

        # get current strains and stresses
        self.project_sig1_ten(self.q_sig1_ten) # get stress component

        epsv_list = self.q_epsv.vector().get_local()  # old visco strains (=deviatoric part)
        sig1_list = self.q_sig1_ten.vector().get_local()

        # compute visco strain from old one epsv
        self.new_epsv = np.zeros_like(epsv_list)

        if self.p.visco_case.lower() == 'cmaxwell':
            mu_E1 = 0.5 * self.p.E_1 / (1. + self.p.nu)
            factor = 1 + self.dt * 2. * mu_E1 / self.p.eta
            self.new_epsv = 1. / factor * (epsv_list + self.dt/self.p.eta * sig1_list)
        elif self.p.visco_case.lower() == 'ckelvin':
            mu_E1 = 0.5 * self.p.E_1 / (1. + self.p.nu)
            mu_E0 = 0.5 * self.p.E_0 / (1. + self.p.nu)
            factor = 1 + self.dt * 2. * mu_E0 / self.p.eta + self.dt * 2. * mu_E1 / self.p.eta
            self.new_epsv = 1. / factor * (epsv_list + self.dt / self.p.eta * sig1_list)
        else:
            raise ValueError('visco case not defined')

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

        sigma_plot = df.project(self.q_sig, self.visu_space_V,
                                form_compiler_parameters={'quadrature_degree': self.p.degree})
        eps_plot = df.project(self.q_eps, self.visu_space_V,
                              form_compiler_parameters={'quadrature_degree': self.p.degree})
        E_plot = df.project(self.q_E, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})
        pd_plot = df.project(self.q_pd, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})

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
    # with the option of time dependent parameters E_0(t), E_1(t), eta(t)

    def __init__(self, mesh, p, pv_name='mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
            raise ValueError('Material law not implemented for 1D')
        elif self.p.dim == 2:
            self.stress_vector_dim = 3
        elif self.p.dim == 3:
            self.stress_vector_dim = 6

        if mesh != None:
            # initialize possible paraview output
            self.pv_file = df.XDMFFile(pv_name + '.xdmf')
            self.pv_file.parameters["flush_output"] = True
            self.pv_file.parameters["functions_share_mesh"] = True
            # function space for single value per element, required for plot of quadrature space values

            #
            if self.p.degree == 1:
                self.visu_space = df.FunctionSpace(mesh, "DG", 0)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)
                self.visu_space_V = df.VectorFunctionSpace(mesh, 'DG', 0,
                                                           dim=self.stress_vector_dim)  # visu space for sigma and eps in voigt notation
            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)
                self.visu_space_V = df.VectorFunctionSpace(mesh, 'P', 1,
                                                           dim=self.stress_vector_dim)  # visu space for sigma and eps in voigt notation

            metadata = {"quadrature_degree": self.p.degree, "quadrature_scheme": "default"}
            dxm = df.dx(metadata=metadata)

            # solution field
            self.V = df.VectorFunctionSpace(mesh, 'P', self.p.degree)

            # generic quadrature function space
            cell = mesh.ufl_cell()
            q = "Quadrature"

            quadrature_element = df.FiniteElement(q, cell, degree=self.p.degree, quad_scheme="default")
            quadrature_vector_element = df.TensorElement(q, cell, degree=self.p.degree, quad_scheme="default")
            quadrature_vector_element01 = df.VectorElement(q, cell, degree=self.p.degree, dim=self.stress_vector_dim,
                                                         quad_scheme="default")
            q_V = df.FunctionSpace(mesh, quadrature_element)
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element) # full tensor
            q_VTV = df.FunctionSpace(mesh, quadrature_vector_element01) # voigt notation

            # quadrature functions
            # to initialize values (otherwise initialized by 0)
            self.q_path = df.Function(q_V, name="path time defined overall")  # negative values where not active yet

            # computed values
            self.q_pd = df.Function(q_V, name="pseudo density")  # active or nonactive
            self.q_E = df.Function(q_V, name="elastic youngs modulus")
            self.q_epsv = df.Function(q_VT, name='visco strain') # full tensor
            self.q_sig1_ten = df.Function(q_VT, name='tensor strain') # full tensor
            self.q_eps = df.Function(q_VTV, name='total strain') # voigt notation
            self.q_sig = df.Function(q_VTV, name='total stress') # voigt notation

            # Define variational problem
            self.u = df.Function(self.V)  # full displacement
            v = df.TestFunction(self.V)

            # Volume force ??? correct?
            if self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # multiplication with activated elements
            R_ufl = df.inner(self.sigma(self.u), self.eps(v)) * dxm  # part with eps
            R_ufl += - self.q_pd * df.inner(self.sigma_2(), self.eps(v)) * dxm  # visco part
            R_ufl += - self.q_pd * df.inner(f, v) * dxm  # add volumetric force, aka gravity (in this case)

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.u)
            # quadrature part
            self.dR = dR_ufl

            # stress and strain projection methods
            self.project_sigma = LocalProjector(self.sigma_voigt(self.sigma(self.u) - self.sigma_2()), q_VTV, dxm)
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VTV, dxm)
            self.project_sig1_ten = LocalProjector(self.sigma_1(self.u), q_VT, dxm) # stress component for visco strain computation

            self.assembler = None  # set as default, to check if bc have been added???

    def sigma(self, v): #total stress without visco part
        mu_E0 = self.p.E_0 / (2.0 * (1.0 + self.p.nu))
        lmb_E0 = self.p.E_0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

        if self.p.dim == 2 and self.p.stress_case == 'plane_stress':
            lmb_E0 = 2 * mu_E0 * lmb_E0 / (
                        lmb_E0 + 2 * mu_E0)  # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html

        if self.p.visco_case.lower() == 'cmaxwell':
            sig = 2.0 * mu_E0 * self.eps(v) + lmb_E0 * df.tr(self.eps(v)) * df.Identity(self.p.dim) + self.sigma_1(v) # stress stiffness zero + stress stiffness one
        elif self.p.visco_case.lower() == 'ckelvin':
            sig = 2.0 * mu_E0 * self.eps(v) + lmb_E0 * df.tr(self.eps(v)) * df.Identity(self.p.dim) # stress stiffness zero
        else:
            sig = None
            raise ValueError('case not defined')


        return sig

    def sigma_1(self, v): #stress stiffness one
        if self.p.visco_case.lower() == 'cmaxwell':
            mu_E1 = self.p.E_1 / (2.0 * (1.0 + self.p.nu))
            lmb_E1 = self.p.E_1 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
            if self.p.dim ==2 and self.p.stress_case == 'plane_stress':
                lmb_E1 = 2 * mu_E1 * lmb_E1 / (lmb_E1 + 2 * mu_E1)
            sig1 = 2.0 * mu_E1 * self.eps(v) + lmb_E1 * df.tr(self.eps(v)) * df.Identity(self.p.dim)
        elif self.p.visco_case.lower() == 'ckelvin':
            sig1 = self.sigma(v)
        else:
            sig = None
            raise ValueError('case not defined')

        return sig1

    def sigma_2(self):  # related to epsv
        if self.p.visco_case.lower() == 'cmaxwell':
            mu_E1 = self.p.E_1 / (2.0 * (1. + self.p.nu))
            sig2 = 2 * mu_E1 * self.q_epsv
        elif self.p.visco_case.lower() == 'ckelvin':
            mu_E0 = self.p.E_0 / (2.0 * (1. + self.p.nu))
            sig2 = 2 * mu_E0 * self.q_epsv
        else:
            sig = None
            raise ValueError('case not defined')
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
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], eT[2, 2], 2 * eT[0, 1], 2 * eT[1, 2], 2 * eT[0, 2]))

        return strain_vector

    def sigma_voigt(self, s):
        # 2D option
        if s.ufl_shape == (2, 2):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[0, 1]))
        # 3D option
        elif s.ufl_shape == (3, 3):
            stress_vector = df.as_vector((s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]))
        else:
            raise ('Problem with stress tensor shape for voigt notation')

        return stress_vector

    def E_fkt(self, pd, path_time, parameters):

        if pd > 0:  # element active, compute current Young's modulus
            age = parameters['age_0'] + path_time  # age concrete
            if age < parameters['t_f'][0]:
                E0 = parameters['E_0'] + parameters['R_i'][0] * age
                E1 = parameters['E_1'] + parameters['R_i'][1] * age
                eta = parameters['E_1'] + parameters['R_i'][2] * age
            elif age >= parameters['t_f'][0]:
                E0 = parameters['E_0'] + parameters['R_i'][0] * parameters['t_f'][0] + parameters['A_i'][0] * (
                            age - parameters['t_f'][0])
                E1 = parameters['E_1'] + parameters['R_i'][1] * parameters['t_f'][1] + parameters['A_i'][1] * (
                        age - parameters['t_f'][1])
                eta = parameters['E_1'] + parameters['R_i'][2] * parameters['t_f'][2] + parameters['A_i'][2] * (
                        age - parameters['t_f'][2])
        else:
            E0 = df.DOLFIN_EPS  # non-active
            E1 = df.DOLFIN_EPS  # non-active
            eta = df.DOLFIN_EPS  # non-active

        return E0, E1, eta

    def pd_fkt(self, path_time):
        # pseudo denisty: decide if layer is active or not (age < 0 nonactive!)
        # decision based on current path_time value
        l_active = 0  # non-active
        if path_time >= 0 - df.DOLFIN_EPS:
            l_active = 1.0  # active
        return l_active

    def evaluate_material(self):
        # get path time; convert quadrature spaces to numpy vector
        path_list = self.q_path.vector().get_local()
        # print('check', path_list)
        # vectorize the function for speed up
        pd_fkt_vectorized = np.vectorize(self.pd_fkt)
        pd_list = pd_fkt_vectorized(path_list)  # current pseudo density 1 if path_time >=0 else 0
        # print('pseudo density', pd_list.max(), pd_list.min())

        # compute current Young's modulus
        parameters = {}
        parameters['E_0'] = self.p.E_0
        #
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, parameters)
        # print('E',E_list.max(),E_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)

        # get current strains and stresses
        self.project_sig1_ten(self.q_sig1_ten) # get stress component

        epsv_list = self.q_epsv.vector().get_local()  # old visco strains (=deviatoric part)
        sig1_list = self.q_sig1_ten.vector().get_local()

        # compute visco strain from old one epsv
        self.new_epsv = np.zeros_like(epsv_list)

        if self.p.visco_case.lower() == 'cmaxwell':
            mu_E1 = 0.5 * self.p.E_1 / (1. + self.p.nu)
            factor = 1 + self.dt * 2. * mu_E1 / self.p.eta
            self.new_epsv = 1. / factor * (epsv_list + self.dt/self.p.eta * sig1_list)
        elif self.p.visco_case.lower() == 'ckelvin':
            mu_E1 = 0.5 * self.p.E_1 / (1. + self.p.nu)
            mu_E0 = 0.5 * self.p.E_0 / (1. + self.p.nu)
            factor = 1 + self.dt * 2. * mu_E0 / self.p.eta + self.dt * 2. * mu_E1 / self.p.eta
            self.new_epsv = 1. / factor * (epsv_list + self.dt / self.p.eta * sig1_list)
        else:
            raise ValueError('visco case not defined')

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

        sigma_plot = df.project(self.q_sig, self.visu_space_V,
                                form_compiler_parameters={'quadrature_degree': self.p.degree})
        eps_plot = df.project(self.q_eps, self.visu_space_V,
                              form_compiler_parameters={'quadrature_degree': self.p.degree})
        E_plot = df.project(self.q_E, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})
        pd_plot = df.project(self.q_pd, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})

        E_plot.rename("Young's Modulus", "Young's modulus value")
        pd_plot.rename("pseudo density", "pseudo density")
        sigma_plot.rename("Stress", "stress components voigt")
        eps_plot.rename("strain", "strain components voigt")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(eps_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
