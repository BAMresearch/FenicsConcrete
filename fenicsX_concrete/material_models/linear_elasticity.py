#import sys
#print(sys.path)
import dolfinx as df
import ufl
from petsc4py.PETSc import ScalarType
#from fenicsX_concrete.randomfieldModified import Randomfield
from fenicsX_concrete.material_models.material import MaterialProblem
from fenicsX_concrete.helpers import Parameters


# this is necessary, otherwise this warning will not stop
# https://fenics.readthedocs.io/projects/ffc/en/latest/_modules/ffc/quadrature/deprecation.html
import warnings
#from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
#df.parameters["form_compiler"]["representation"] = "quadrature"
#warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


class LinearElasticity(MaterialProblem):
    """Material definition for linear elasticity"""

    def __init__(self, experiment=None, parameters=None, pv_name='pv_output_linear_elasticity'):
        """Initializes the object by calling super().__init__

        Parameters
        ----------
            experiment : object, optional
                When no experiment is passed, the dummy experiment "MinimalCubeExperiment" is added
            parameters : dictionary, optional
                Dictionary with parameters. When none is provided, default values are used
            pv_name : string, optional
                Name of the paraview file, if paraview output is generated
        """
        # generate "dummy" experiment when none is passed

        #if experiment is None:
        #    #experiment = experimental_setups.MinimalCubeExperiment(parameters)

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        default_p = Parameters()

        # parameters for mechanics problem
        default_p['E'] = None  # Young's Modulus
        default_p['nu'] = None  # Poisson's Ratio
        default_p['mu'] = None
        default_p['lmbda'] = None

        self.p = default_p + self.p

        # expecting E and nu to compute mu and lambda, however you can directly supply mu and lambda
        # compute material parameters
        
        #if self.p.mu is None or self.p.lmbda is None:
        #    assert self.p.E is not None and self.p.nu is not None
        #    self.p.mu = self.p.E / (2.0 * (1.0 + self.p.nu))
        #    self.p.lmbda = self.p.E * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

        # initialize possible paraview output
        #self.pv_file = df.io.XDMFFile(self.experiment.mesh.comm, self.pv_name + '.xdmf',  "w")
        #self.pv_file.parameters["flush_output"] = True
        #self.pv_file.parameters["functions_share_mesh"] = True

        self.residual = None  # initialize residual

        #T = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, 0)))
     
        #ds = ufl.Measure("ds", domain=self.experiment.mesh)
        #self.L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
        #self.L = df.inner(f, v) * df.dx                          # older version

        ##Probabilistic
        ## Creating random fields for E (Young's modulus) and Mu (Poisson's ratio) constants.
#
        #def random_field_generator(domain, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values, ktol):
        #    field_function_space = df.fem.FunctionSpace(domain, ("CG", 1))
        #    random_field = Randomfield(field_function_space, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values, ktol)
        #    #random_field = Randomfield(fct_space=var_function_space, cov_name='squared_exp', mean=1, rho=0.5, sigma=1, k=10)
        #    #random_field.solve_covariance_EVP()
        #    return random_field
#
        #def parameters_conversion(lognormal_mean, lognormal_sigma):
        #    from numpy import log
        #    from math import sqrt
        #    normal_mean = log(lognormal_mean/sqrt(1 + (lognormal_sigma/lognormal_mean)**2))
        #    normal_sigma = log(1 + (lognormal_sigma/lognormal_mean)**2)
        #    return normal_mean, normal_sigma
#
        #E_mean, E_variance = parameters_conversion(self.p.E, 3)#3
        #Nu_mean, Nu_variance = parameters_conversion(self.p.nu, 0.03)#0.03
#
        #            # print(E_mean, E_variance, Nu_mean, Nu_variance)
        #            # Deterministic self.p.E changes to random field version here!
#
        #self.p.E  = random_field_generator(self.experiment.mesh,'squared_exp', E_mean, 0.3, 0.05, E_variance, 3, 0.01) 
        #self.p.E.create_random_field(_type='random', _dist='LN')
#
        #self.p.nu = random_field_generator(self.experiment.mesh,'squared_exp', Nu_mean, 0.3, 0.05, Nu_variance, 3, 0.01)
        #self.p.nu.create_random_field(_type='random', _dist='LN')
#
        #            # displacement field
        #            #self.displacement = df.Function(self.V)

        # TODO better names!!!!
        #self.visu_space_T = df.TensorFunctionSpace(self.experiment.mesh, "Lagrange", self.p.degree)

        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.experiment.V)
        self.v = ufl.TestFunction(self.experiment.V)
        
        if self.p.dim == 2:
            #f = df.Constant((0, 0))
            f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) 
        elif self.p.dim == 3:
            #f = df.Constant((0, 0, 0))
            f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
        else:
            raise Exception(f'wrong dimension {self.p.dim} for problem setup')
        
        self.T = df.fem.Constant(self.experiment.mesh, ScalarType((10, 0)))
        ds = self.experiment.create_neumann_boundary()
        self.L =  ufl.dot(self.T, self.v) * ds(1) #+ ufl.dot(f, self.v) * ufl.dx

        #self.E = df.fem.Constant(self.experiment.mesh, 100.)
        #self.nu = df.fem.Constant(self.experiment.mesh, 0.2)
        
        E = 100
        nu = 0.2

        self.lambda_ = df.fem.Constant(self.experiment.mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
        self.mu = df.fem.Constant(self.experiment.mesh, E / (2.0 * (1.0 + nu)))

        #self.lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        #self.mu = E / (2.0 * (1.0 + nu))

        self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx

        self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 

    #Probablistic
    #def sigma(self, u):
    #    return self.lambda_() * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu()*self.epsilon(u)

    #Deterministic
    def sigma(self, u):
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu*self.epsilon(u)

    def solve(self, t=1.0):
        #Deterministic
        #self.lambda_ = self.p.E * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
        #self.mu = self.p.E / (2.0 * (1.0 + self.p.nu))

        #self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx
        
        # time in this example only relevant for the naming of the paraview steps and the sensor output
        # solve
        # weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        # self.displacement = weak_form_problem.solve()

        self.displacement = self.weak_form_problem.solve()
        #print(self.displacement.x.array.shape)
        #print(self.displacement._V.tabulate_dof_coordinates().shape)
        #print(self.displacement.x.array)

        #df.solve(self.a == self.L, self.displacement, self.bcs)

        self.stress = self.sigma(self.displacement)

        # TODO make some switch in sensor definition to trigger this...
        #self.compute_residual()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)
            
    #def compute_residual(self):
    #    # compute reaction forces
    #    self.residual = df.action(self.a, self.displacement) - self.L

    def pv_plot(self, t=0):
        # paraview output

        # displacement plot
        #u_plot = df.project(self.displacement, self.V)
        #u_plot.rename("Displacement", "test string, what does this do??")  # TODO: what does the second string do?
        #self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        #
        ## stress plot
        #sigma_plot = df.project(self.stress, self.visu_space_T)
        #sigma_plot.rename("Stress", "test string, what does this do??")  # TODO: what does the second string do?
        #self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        
        # Displacement Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Displacement.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh)
            xdmf.write_function(self.displacement)

        #Stress Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Stress.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh.comm)
        #    xdmf.write_function(self.stress)

