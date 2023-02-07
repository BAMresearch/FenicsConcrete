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

        for i in self.p['uncertainties']:
            if i == 1:
                print(i)
          
        self.residual = None  # initialize residual

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

        #T = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, 0)))
     
        #ds = ufl.Measure("ds", domain=self.experiment.mesh)
        #self.L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
        #self.L = df.inner(f, v) * df.dx                          # older version

        # print(E_mean, E_variance, Nu_mean, Nu_variance)
        # Deterministic self.p.E changes to random field version here!

        # Probabilistic
        # Creating random fields for E (Young's modulus) and Mu (Poisson's ratio) constants.

        # Constant E and nu fields.
        E = self.p.E 
        nu = self.p.nu 
        
        self.lambda_ = df.fem.Constant(self.experiment.mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
        self.mu = df.fem.Constant(self.experiment.mesh, E / (2.0 * (1.0 + nu)))

        ## Random E and nu fields.
        #def random_field_generator(field_function_space, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values, ktol):
        #    random_field = Randomfield(field_function_space, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values, ktol)
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
        #self.field_function_space = df.fem.FunctionSpace(self.experiment.mesh, ("CG", 1))
        #self.p.E  = random_field_generator(self.field_function_space,'squared_exp', E_mean, 0.3, 0.05, E_variance, 3, 0.01) 
        #self.p.E.create_random_field(_type='random', _dist='LN')
#
        #self.p.nu = random_field_generator(self.field_function_space,'squared_exp', Nu_mean, 0.3, 0.05, Nu_variance, 3, 0.01)
        #self.p.nu.create_random_field(_type='random', _dist='LN')
#
        #self.lame1 = (self.p.E.field.vector[:] * self.p.nu.field.vector[:])/((1 + self.p.nu.field.vector[:])*(1-2*self.p.nu.field.vector[:]))
        #self.lame2 = self.p.E.field.vector[:]/(2*(1+self.p.nu.field.vector[:]))
#
        #self.lambda_ = df.fem.Function(self.field_function_space)
        #self.mu = df.fem.Function(self.field_function_space)
#
        #self.lambda_.vector[:] = self.lame1
        #self.mu.vector[:] = self.lame2   

        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.experiment.V)
        self.v = ufl.TestFunction(self.experiment.V)
        
        #self.T = df.fem.Constant(self.experiment.mesh, ScalarType((10, 0)))

        # Selects the problem which you want to solve
        if self.p.problem == 'tensile_test':
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load, 0)))
            ds = self.experiment.create_neumann_boundary()
            self.L =  ufl.dot(self.T, self.v) * ds(1) 

        elif self.p.problem == 'cantilever_beam':
            if self.p.dim == 2:
                #f = df.Constant((0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) 
            elif self.p.dim == 3:
                #f = df.Constant((0, 0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
            else:
                raise Exception(f'wrong dimension {self.p.dim} for problem setup')
                
            self.L =  ufl.dot(f, self.v) * ufl.dx
        else:
            exit()
        
        K_torsion = self.p.K_torsion
        self.K_torsion = df.fem.Constant(self.experiment.mesh, K_torsion)
        
        moment_arm = df.fem.Function(self.experiment.V_scalar)

        import numpy as np
        def moment_arm_(x):
            a=np.unique(x[1])
            length_x = x[1].shape[0]
            moment_arm_array = np.zeros(length_x)
            transformed_coordinates_vector = np.absolute(x[1]) - 0.5*self.p.breadth
            b=np.unique(x[1])
            for i in range(length_x):
                if transformed_coordinates_vector[i]!=0:
                    moment_arm_array[i] = 1/transformed_coordinates_vector[i]**2
            return moment_arm_array
    
        moment_arm.interpolate(moment_arm_)
        #moment_arm.interpolate(lambda x: print(type(x),x))
        #moment_arm.interpolate(lambda x: print(type(x[1])))#1/x[1]**2)
        
        
        #comparr = 0.5*self.p.breadth*np.ones(2400)
        #moment_arm.interpolate(lambda x : 1/(self.p.breadth-x[1])**2 if x[1]>=0.5*self.p.breadth else 1/x[1]**2)
        #moment_arm.interpolate(lambda x : 1/(x[1])**2 if x[1]>=comparr else x[1])
        spring_stiffness = ufl.as_matrix([[self.K_torsion*moment_arm, 0], [0, 0]])
        self.spring_stress = ufl.dot(spring_stiffness,self.u_trial)

        def clamped_boundary_in_y(x):
            return np.isclose(x[0], 0)
        
        def clamped_neutral_axis(x):          
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1],0.1))
            #return np.isclose(x[0], 0)

        fdim =  0
        #bc = df.fem.dirichletbc(ScalarType(0), df.fem.locate_dofs_topological(self.experiment.V.sub(1), fdim, boundary_facets), self.experiment.V.sub(1))

        #Way1
        #boundary_facets = df.mesh.locate_entities_boundary(self.experiment.mesh, fdim, clamped_neutral_axis)
        #sub0 = df.fem.locate_dofs_topological(self.experiment.V.sub(0), fdim, boundary_facets)
        #sub1 = df.fem.locate_dofs_topological(self.experiment.V.sub(1), fdim, boundary_facets)
        #sub = df.fem.locate_dofs_topological(self.experiment.V, fdim, boundary_facets)

        #bc1 = df.fem.dirichletbc(ScalarType(0), sub0, self.experiment.V.sub(0))
        #bc2 = df.fem.dirichletbc(ScalarType(0), sub1, self.experiment.V.sub(1))


        #Way2
        #dofs_geo = df.fem.locate_dofs_geometrical(self.experiment.V, clamped_neutral_axis)
        #bc1 = df.fem.dirichletbc(ScalarType(0), dofs_geo, self.experiment.V.sub(0))
        #bc2 = df.fem.dirichletbc(ScalarType(0), dofs_geo, self.experiment.V.sub(1))

        #Way3
        #dofs_geo = df.fem.locate_dofs_geometrical(self.experiment.V, clamped_neutral_axis)
        #bc2 = df.fem.dirichletbc(ScalarType(0), dofs_geo, self.experiment.V.sub(0))

        bc1_facets = df.mesh.locate_entities_boundary(self.experiment.mesh, 0, clamped_neutral_axis)
        bc1_dofs_sub1 = df.fem.locate_dofs_topological(self.experiment.V.sub(1), 0, bc1_facets)
        bc1 = df.fem.dirichletbc(ScalarType(0), bc1_dofs_sub1, self.experiment.V.sub(1))

        bc2_facets = df.mesh.locate_entities_boundary(self.experiment.mesh, 0, clamped_neutral_axis)
        bc2_dofs_sub0 = df.fem.locate_dofs_topological(self.experiment.V.sub(0), 0, bc2_facets)
        bc2 = df.fem.dirichletbc(ScalarType(0), bc2_dofs_sub0, self.experiment.V.sub(0))


        ds = self.experiment.create_neumann_boundary()      


        #ufl.Identity(self.u_trial.geometric_dimension())

        self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx + ufl.dot(self.spring_stress, self.v) * ds(2) 
        #self.experiment.bcs,
        self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    
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

        self.displacement = self.weak_form_problem.solve()
        #print(self.displacement.x.array.shape)
        #print(self.displacement._V.tabulate_dof_coordinates().shape)
        #print(self.displacement.x.array)

        #self.stress = self.sigma(self.displacement)

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
        
        # Displacement Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Displacement.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh)
            xdmf.write_function(self.displacement)

        #Stress Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Stress.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh.comm)
        #    xdmf.write_function(self.stress)

        ## Youngs Modulus Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Youngs_Modulus.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.p.E.field)
        #    #xdmf.write_function(self.p.nu.field)
#
        ## Poissons ratio Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Poissons_Ratio.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.p.nu.field)

