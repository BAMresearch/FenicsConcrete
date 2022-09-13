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

class ConcreteViscoMechanical(MaterialProblem):
    def __init__(self, experiment=None, parameters=None, pv_name='pv_output_concrete-visco'):
        # generate "dummy" experiment when none is passed
        if experiment == None:
            experiment = experimental_setups.get_experiment('MinimalCube', parameters)

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        # setup initial material parameters
        default_p = Parameters()
        # Material parameter for concrete model with viscoelasticity default values
        default_p['density'] = 2070  # in kg/m^3 density of fresh concrete
        # polynomial degree
        default_p['degree'] = 1  # default boundary setting
        ### paramters for mechanics problem
        default_p['nu'] = 0.3  # Poissons Ratio
        default_p['E0'] = 40000  # Youngs Modulus Pa linear elastic
        default_p['E1'] = 20000  # Youngs Modulus Pa visco element
        default_p['eta'] = 1000   # Damping coeff

        self.p = default_p + self.p

        # create model
        self.mechanics_problem = ConcreteViscoElasticModel(self.experiment.mesh, self.p, pv_name=self.pv_name)
        self.V = self.mechanics_problem.V  # for reaction force sensor
        self.residual = None  # initialize

        # setting bcs
        bcs = self.experiment.create_displ_bcs(self.mechanics_problem.V)  # fixed boundary bottom

        self.mechanics_problem.set_bcs(bcs)

        # setting up the solver
        self.mechanics_solver = df.NewtonSolver()
        self.mechanics_solver.parameters['absolute_tolerance'] = 1e-8
        self.mechanics_solver.parameters['relative_tolerance'] = 1e-8

    def set_initial_path(self, path):
        self.mechanics_problem.set_initial_path(path)

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
        self.residual = self.mechanics_problem.R  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

        # update before next step!
        self.mechanics_problem.update_values()

    def pv_plot(self, t=0):
        # calls paraview output for both problems
        self.mechanics_problem.pv_plot(t=t)

    def set_timestep(self, dt):
        self.mechanics_problem.set_timestep(dt)

class ConcreteViscoElasticModel(df.NonlinearProblem):
    def __init__(self, mesh, p, pv_name='mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self)  # apparently required to initialize things
        self.p = p

        if self.p.dim == 1:
            self.stress_vector_dim = 1
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
            else:
                self.visu_space = df.FunctionSpace(mesh, "P", 1)
                self.visu_space_T = df.TensorFunctionSpace(mesh, "P", 1)

            metadata = {"quadrature_degree": self.qd, "quadrature_scheme": "default"}
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

            # Define variational problem
            self.u = df.Function(self.V)  # full displacement
            v = df.TestFunction(self.V)

            # Volume force ??? correct?
            if self.p.dim == 1:
                f = df.Constant(-self.p.g * self.p.density)
            elif self.p.dim == 2:
                f = df.Constant((0, -self.p.g * self.p.density))
            elif self.p.dim == 3:
                f = df.Constant((0, 0, -self.p.g * self.p.density))

            # define sigma in evaluate material or here globally
            self.mu_0 = 0.5 * self.p['E0'] / (1 + self.p['nu'])
            self.lmbda_0 = self.p['E0'] * self.p['nu'] / ((1 - 2 * self.p['nu']) * (1 + self.p['nu']))
            self.mu_1 = 0.5 * self.p['E1'] / (1 + self.p['nu'])  # 2nd Lame constant
            self.lmbda_1 = self.p['E1'] * self.p['nu'] / ((1 - 2 * self.p['nu']) * (1 + self.p['nu']))

            self.sigma_ufl = self.sigma_el(self.u)-self.sigma_vi() # C_1 *eps + C_2 * (eps-epsv)

            # multiplication with activated elements
            R_ufl = self.q_E * df.inner(self.sigma_1(self.u), self.eps(v)) * dxm # part with eps
            R_ufl += - self.q_pd * df.inner(self.sigma_2(), self.eps(v)) * dxm  # visco part
            R_ufl += - self.q_pd * df.inner(f, v) * dxm  # add volumetric force, aka gravity (in this case)

            # quadrature point part
            self.R = R_ufl

            # derivative
            # normal form
            dR_ufl = df.derivative(R_ufl, self.u)
            # quadrature part
            self.dR = dR_ufl

            # stress and strain
            self.project_sigma = LocalProjector(self.sigma_voigt(self.sigma_ufl), q_VT, dxm)
            self.project_strain = LocalProjector(self.eps_voigt(self.u), q_VT, dxm)

            self.assembler = None  # set as default, to check if bc have been added???

    def sigma_1(self, v):  # related to eps
        return self.dotC(self.mu_0, self.lmbda_0) * self.eps(v) + self.dotC(self.mu_1, self.lmbda_1) * self.eps(v)

    def sigma_2(self):  # related to epsv
        return self.dotC(self.mu_1, self.lmbda_1) * self.q_epsv

    def eps(self,v):
        return df.sym(df.grad(v))

    def dotC(self, mu, lmbda):  # Elastic tensor
        C = df.as_matrix([[2 * mu + lmbda, lmbda, 0],
                       [lmbda, 2 * mu + lmbda, 0],
                       [0, 0, mu]])
        return C

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

    def E_fkt(self, pd, path_time, parameters):

        if pd > 0: # element active, compute current Young's modulus
            E = parameters['E0'] # no thixotropy evaluation yet!!!
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
        parameters['E_0'] = self.p.E0
        #
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(pd_list, path_list, parameters)
        # print('E',E_list.max(),E_list.min())

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        set_q(self.q_pd, pd_list)

    def update_values(self):
        # update process time
        path_list = self.q_path.vector().get_local()
        path_list += self.dt * np.ones_like(path_list)

        set_q(self.q_path, path_list)

        # update visco elastic strain and u
        eps_list = self.project_strain.vector().get_local() # strains time n
        epsv_list = self.q_epsv.vector().get_local() # visco strains time n

        factor = 1+self.dt*self.p.E1/self.p.eta
        new_epsv = 1/factor * (epsv_list + self.dt * self.p.E1 * eps_list) # visco strain n+1

        set_q(self.q_epsv, new_epsv)

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

        # additional eps_v
        epsv_plot = df.project(self.q_epsv, self.visu_space_T, form_compiler_parameters={'quadrature_degree': self.p.degree})

        E_plot.rename("Young's Modulus", "Young's modulus value")
        sigma_plot.rename("Stress", "stress components")
        pd_plot.rename("pseudo density", "pseudo density")
        epsv_plot.rename("visco strain", "visco strain")

        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(pd_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(epsv_plot, t, encoding=df.XDMFFile.Encoding.ASCII)


