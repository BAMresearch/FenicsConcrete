#import sys
#print(sys.path)
import dolfinx as df
import ufl
from petsc4py.PETSc import ScalarType
from fenicsX_concrete.randomfieldModified import Randomfield
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

        # Creating random fields for E (Young's modulus) and Mu (Poisson's ratio) constants.

        def random_field_generator(domain, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values):
            field_function_space = df.fem.FunctionSpace(domain, ("CG", 1))
            random_field = Randomfield(field_function_space, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values)
            #random_field = Randomfield(fct_space=var_function_space, cov_name='squared_exp', mean=1, rho=0.5, sigma=1, k=10)
            #random_field.solve_covariance_EVP()
            return random_field

        self.p.E  = random_field_generator(self.experiment.mesh,'squared_exp', 100, 0.3, 0.05, 0, 3) 
        self.p.E.create_random_field(_type='random')

        self.p.nu = random_field_generator(self.experiment.mesh,'squared_exp', 0.2, 0.3, 0.05, 0, 3)
        self.p.nu.create_random_field(_type='random')

        # displacement field
        #self.displacement = df.Function(self.V)

        # TODO better names!!!!
        #self.visu_space_T = df.TensorFunctionSpace(self.experiment.mesh, "Lagrange", self.p.degree)

        # Define variational problem
        u_trial = ufl.TrialFunction(self.experiment.V)
        v = ufl.TestFunction(self.experiment.V)
        #self.a = ufl.inner(self.sigma(u_trial), df.grad(v)) * df.dx

        if self.p.dim == 2:
            #f = df.Constant((0, 0))
            f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) 
        elif self.p.dim == 3:
            #f = df.Constant((0, 0, 0))
            f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
        else:
            raise Exception(f'wrong dimension {self.p.dim} for problem setup')
        self.a = ufl.inner(self.sigma(u_trial), self.epsilon(v)) * ufl.dx
        self.L = ufl.dot(f, v) * ufl.dx

    # Stress computation for linear elastic problem

    def lambda_(self): #Lame's constant
        return (self.p.E.field * self.p.nu.field)/((1 + self.p.E.field)*(1-2*self.p.nu.field))

    def mu(self):     #Lame's constant
        return self.p.E.field/(2*(1+self.p.nu.field))

    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 

    def sigma(self, u):
        return self.lambda_() * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu()*self.epsilon(u)


    #def sigma(self, v):
    #    # v is the displacement field
    #    return 2.0 * self.p.mu * df.sym(df.grad(v)) + self.p.lmbda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))

    def solve(self, t=1.0):
        # time in this example only relevant for the naming of the paraview steps and the sensor output
        # solve
        weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.displacement = weak_form_problem.solve()
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
            xdmf.write_mesh(self.experiment.mesh.comm)
            xdmf.write_function(self.displacement)

        #Stress Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Stress.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh.comm)
            xdmf.write_function(self.stress)

