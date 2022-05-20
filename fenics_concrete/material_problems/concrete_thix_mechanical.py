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

# copy from concrete_thermo_mechanical.py
# change/ adapted models for modelling structural build-up
class ConcreteThixMechanical(MaterialProblem):
    def __init__(self, experiment=None, parameters=None, pv_name='pv_output_concrete-thix'):
        # generate "dummy" experiment when none is passed
        if experiment == None:
            experiment = experimental_setups.get_experiment('MinimalCube', parameters)

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        # setup initial material parameters
        default_p = Parameters()
        # Material parameter for concrete model with structural build-up
        default_p['density'] = 2070  # in kg/m^3 density of fresh concrete see Wolfs et al 2018
        default_p['u_bc'] = 0.1 # displacement on top

        # temperature dependency on structural build-up not yet included
        default_p['T'] = 22 # current ambient temperature in degree celsius
        default_p['T_ref'] = 25  # reference ambient temperature in degree celsius

        # polynomial degree
        default_p['degree'] = 2  # default boundary setting

        ### paramters for mechanics problem
        default_p['nu'] = 0.3       # Poissons Ratio see Wolfs et al 2018
        default_p['E_0'] = 15000    # Youngs Modulus Pa # random values!!
        default_p['R_E'] = 15       # reflocculation rate of E modulus in Pa / s
        default_p['A_E'] = 30       # structuration rate of E modulus in Pa / s
        default_p['t_f'] = 300      # reflocculation time in s

        self.p = default_p + self.p

        # create model
        self.mechanics_problem = ConcreteThixElasticModel(self.experiment.mesh, self.p, pv_name=self.pv_name)

        # setting bcs
        bcs = self.experiment.create_displ_bcs(self.mechanics_problem.V) # fixed boundary bottom

        self.mechanics_problem.set_bcs(bcs)

        # setting up the solver
        self.mechanics_solver = df.NewtonSolver()
        self.mechanics_solver.parameters['absolute_tolerance'] = 1e-8
        self.mechanics_solver.parameters['relative_tolerance'] = 1e-8

    def set_initial_age(self, age):
        self.mechanics_problem.set_initial_age(age)

    def solve(self, t=1.0):

        # print('solve for',t)
        self.mechanics_solver.solve(self.mechanics_problem, self.mechanics_problem.u.vector())

        # save fields to global problem for sensor output
        self.displacement = self.mechanics_problem.u
        self.stress = self.mechanics_problem.sigma_ufl
        self.strain = self.mechanics_problem.eps(self.mechanics_problem.u)
        self.visu_space_T = self.mechanics_problem.visu_space_T
        # further ?? stress/strain ...

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

        # update age before next step!
        self.mechanics_problem.update_age()

    def pv_plot(self, t=0):
        # calls paraview output for both problems
        self.mechanics_problem.pv_plot(t=t)

    def set_timestep(self, dt):
        self.mechanics_problem.set_timestep(dt)

    def get_E_fkt(self):
        return np.vectorize(self.mechanics_problem.E_fkt)


class ConcreteThixElasticModel(df.NonlinearProblem):
    def __init__(self, mesh, p, pv_name='mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
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
            quadrature_vector_element = df.VectorElement(q, cell, degree=self.p.degree, dim=self.stress_vector_dim,
                                                         quad_scheme="default")
            q_V = df.FunctionSpace(mesh, quadrature_element)
            q_VT = df.FunctionSpace(mesh, quadrature_vector_element)

            # quadrature functions
            self.q_E = df.Function(q_V, name="youngs modulus")
            self.q_age = df.Function(q_V, name="age of concrete")
            self.q_sigma = df.Function(q_VT, name="stress")
            self.q_eps = df.Function(q_VT, name="Strain")

            # Define variational problem
            self.u = df.Function(self.V)  # displacement
            v = df.TestFunction(self.V)

            # Volume force   todo: ANNIKA: dependent on age!
            if self.p.dim == 1:
                f = df.Constant(-self.p.g * self.p.density)
            elif self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # define sigma from(u,t) in evalute material or here global E change ? (see damage example Thomas) -> then tangent by hand!
            # Elasticity parameters without multiplication with E
            self.x_mu = 1.0 / (2.0 * (1.0 + self.p.nu))
            self.x_lambda = 1.0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
            self.sigma_ufl = self.q_E * self.x_sigma(self.u, self.x_mu, self.x_lambda)

            R_ufl = self.q_E * df.inner(self.x_sigma(self.u, self.x_mu, self.x_lambda), self.eps(v)) * dxm
            R_ufl += - df.inner(f, v) * dxm  # add volumetric force, aka gravity (in this case)
            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.u)
            # quadrature part
            self.dR = dR_ufl

            self.project_sigma = LocalProjector(self.sigma_voigt(self.sigma_ufl), q_VT, dxm)
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VT, dxm)

            self.assembler = None  # set as default, to check if bc have been added???

    def x_sigma(self, v, x_mu, x_lambda):
        #add plane stress plane strain options!!
        return 2.0 * x_mu * df.sym(df.grad(v)) + x_lambda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))

    def eps(self,v):
        return df.sym(df.grad(v))

    def sigma_voigt(self, s):
        # 1D option
        if s.ufl_shape == (1, 1):
            stress_vector = df.as_vector((s[0, 0]))
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
            strain_vector = df.as_vector((eT[0, 0]))
        # 2D option
        elif eT.ufl_shape == (2, 2):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], 2 * eT[0, 1]))
        # 3D option
        elif eT.ufl_shape == (3, 3):
            strain_vector = df.as_vector((eT[0, 0], eT[1, 1], eT[2,2], 2 * eT[0, 1], 2*eT[1,2], 2*eT[0,2]))

        return strain_vector

    def E_fkt(self, age, parameters):

        if age < parameters['t_f']:
            E = parameters['E_0'] + parameters['R_E'] * age
        else:
            E = parameters['E_0'] + parameters['R_E'] * parameters['t_f'] + parameters['A_E'] * (age-parameters['t_f'])
        return E

    # def principal_stress(self, stresses):
    #     # checking type of problem
    #     n = stresses.shape[1]  # number of stress components in stress vector
    #     # finding eigenvalues of symmetric stress tensor
    #     # 1D problem
    #     if n == 1:
    #         principal_stresses = stresses
    #     # 2D problem
    #     elif n == 3:
    #         # the following uses
    #         # lambda**2 - tr(sigma)lambda + det(sigma) = 0, solve for lambda using pq formula
    #         p = - (stresses[:, 0] + stresses[:, 1])
    #         q = stresses[:, 0] * stresses[:, 1] - stresses[:, 2] ** 2
    #
    #         D = p ** 2 / 4 - q  # help varibale
    #         assert np.all(D >= -1.0e-15)  # otherwise problem with imaginary numbers
    #         sqrtD = np.sqrt(D)
    #
    #         eigenvalues_1 = -p / 2.0 + sqrtD
    #         eigenvalues_2 = -p / 2.0 - sqrtD
    #
    #         # strack lists as array
    #         principal_stresses = np.column_stack((eigenvalues_1, eigenvalues_2))
    #
    #         # principal_stress = np.array([ev1p,ev2p])
    #     elif n == 6:
    #         # for a symetric stress vector a b c e f d we need to solve:
    #         # x**3 - x**2(a+b+c) - x(e**2+f**2+d**2-ab-bc-ac) + (abc-ae**2-bf**2-cd**2+2def) = 0, solve for x
    #         principal_stresses = np.empty([len(stresses), 3])
    #         # currently slow solution with loop over all stresses and subsequent numpy function call:
    #         for i, stress in enumerate(stresses):
    #             # convert voigt to tensor, (00,11,22,12,02,01)
    #             stress_tensor = np.zeros((3, 3))
    #             stress_tensor[0][0] = stress[0]
    #             stress_tensor[1][1] = stress[1]
    #             stress_tensor[2][2] = stress[2]
    #             stress_tensor[0][1] = stress[5]
    #             stress_tensor[1][2] = stress[3]
    #             stress_tensor[0][2] = stress[4]
    #             stress_tensor[1][0] = stress[5]
    #             stress_tensor[2][1] = stress[3]
    #             stress_tensor[2][0] = stress[4]
    #             # use numpy for eigenvalues
    #             principal_stress = np.linalg.eigvalsh(stress_tensor)
    #             # sort principal stress from lagest to smallest!!!
    #             principal_stresses[i] = -np.sort(-principal_stress)
    #
    #     return principal_stresses

    # def yield_surface(self, stresses, ft, fc):
    #     # function for approximated yield surface
    #     # first approximation, could be changed if we have numbers/information
    #     fc2 = fc
    #     # pass voigt notation and compute the principal stress
    #     p_stresses = self.principal_stress(stresses)
    #
    #     # get the principle tensile stresses
    #     t_stresses = np.where(p_stresses < 0, 0, p_stresses)
    #
    #     # get dimension of problem, ie. length of list with principal stresses
    #     n = p_stresses.shape[1]
    #     # check case
    #     if n == 1:
    #         # rankine for the tensile region
    #         rk_yield_vals = t_stresses[:, 0] - ft[:]
    #
    #         # invariants for drucker prager yield surface
    #         I1 = stresses[:, 0]
    #         I2 = np.zeros_like(I1)
    #     # 2D problem
    #     elif n == 2:
    #
    #         # rankine for the tensile region
    #         rk_yield_vals = (t_stresses[:, 0] ** 2 + t_stresses[:, 1] ** 2) ** 0.5 - ft[:]
    #
    #         # invariants for drucker prager yield surface
    #         I1 = stresses[:, 0] + stresses[:, 1]
    #         I2 = ((stresses[:, 0] + stresses[:, 1]) ** 2 - ((stresses[:, 0]) ** 2 + (stresses[:, 1]) ** 2)) / 2
    #
    #     # 3D problem
    #     elif n == 3:
    #         # rankine for the tensile region
    #         rk_yield_vals = (t_stresses[:, 0] ** 2 + t_stresses[:, 1] ** 2 + t_stresses[:, 2] ** 2) ** 0.5 - ft[:]
    #
    #         # invariants for drucker prager yield surface
    #         I1 = stresses[:, 0] + stresses[:, 1] + stresses[:, 2]
    #         I2 = ((stresses[:, 0] + stresses[:, 1] + stresses[:, 2]) ** 2 - (
    #                     (stresses[:, 0]) ** 2 + (stresses[:, 1]) ** 2 + (stresses[:, 2]) ** 2)) / 2
    #     else:
    #         raise ('Problem with input to yield surface, the array with stress values has the wrong size ')
    #
    #     J2 = 1 / 3 * I1 ** 2 - I2
    #     beta = (3.0 ** 0.5) * (fc2 - fc) / (2 * fc2 - fc)
    #     Hp = fc2 * fc / ((3.0 ** 0.5) * (2 * fc2 - fc))
    #
    #     dp_yield_vals = beta / 3 * I1 + J2 ** 0.5 - Hp
    #
    #     # TODO: is this "correct", does this make sense? for a compression state, what if rk yield > dp yield???
    #     yield_vals = np.maximum(rk_yield_vals, dp_yield_vals)
    #
    #     return np.asarray(yield_vals)

    def evaluate_material(self):
        # convert quadrature spaces to numpy vector
        age_list = self.q_age.vector().get_local()

        parameters = {}
        parameters['t_f'] = self.p.t_f
        parameters['E_0'] = self.p.E_0
        parameters['R_E'] = self.p.R_E
        parameters['A_E'] = self.p.A_E
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(age_list, parameters)

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)

    def update_age(self):
        # no history field currently
        age_list = self.q_age.vector().get_local()
        age_list += self.dt * np.ones_like(age_list)
        set_q(self.q_age, age_list)

    def set_timestep(self, dt):
        self.dt = dt

    def set_initial_age(self, age):
        self.q_age.interpolate(age) # todo: ANNIKA: What's if this is not given as a dolfin expression?

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
        E_plot = df.project(self.q_E, self.visu_space, form_compiler_parameters={'quadrature_degree': self.p.degree})
        E_plot.rename("Young's Modulus", "Young's modulus value")
        sigma_plot.rename("Stress", "stress components")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
