#import sys
#print(sys.path)
import dolfinx as df
import ufl
from petsc4py.PETSc import ScalarType
from fenicsX_concrete.randomfieldModified import Randomfield
from fenicsX_concrete.material_models.material import MaterialProblem
from fenicsX_concrete.helpers import Parameters
import numpy as np

# this is necessary, otherwise this warning will not stop
# https://fenics.readthedocs.io/projects/ffc/en/latest/_modules/ffc/quadrature/deprecation.html
import warnings
#from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
#df.parameters["form_compiler"]["representation"] = "quadrature"
#warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


class IsotropicLinearElasticity(MaterialProblem):
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

        # Constant E and nu fields.
        if 0 in self.p['uncertainties']:
            E = self.p.E 
            nu = self.p.nu 

            self.lambda_ = df.fem.Constant(self.experiment.mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
            self.mu = df.fem.Constant(self.experiment.mesh, E / (2.0 * (1.0 + nu)))

        # Random E and nu fields.
        if 1 in self.p['uncertainties']:
            self.field_function_space = df.fem.FunctionSpace(self.experiment.mesh, ("CG", 1))
            self.lambda_ = df.fem.Function(self.field_function_space)
            self.mu = df.fem.Function(self.field_function_space)

            lame1, lame2 = self.get_lames_constants()
            self.lambda_.vector[:] = lame1
            self.mu.vector[:] = lame2  #make this vector as a fenics constant array. Update the lame1 and lame2 in each iteration.

        #self.lambda_.vector[:] = self.lame1
        #self.mu.vector[:] = self.lame2  #make this vector as a fenics constant array. Update the lame1 and lame2 in each iteration.
        
        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.experiment.V)
        self.v = ufl.TestFunction(self.experiment.V)

        # Selects the problem which you want to solve
        if self.p.problem == 'tensile_test':
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load[0], self.p.load[1]))) #self.p.load
            ds = self.experiment.create_neumann_boundary()
            self.L =  ufl.dot(self.T, self.v) * ds(1) 

        elif self.p.problem == 'cantilever_beam':
            if self.p.dim == 2:
                #f = df.Constant((0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) #0, -self.p.rho*self.p.g
            elif self.p.dim == 3:
                #f = df.Constant((0, 0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
            else:
                raise Exception(f'wrong dimension {self.p.dim} for problem setup')
                
            self.L =  ufl.dot(f, self.v) * ufl.dx
        else:
            exit()

        #For torsional spring
        def moment_arm_(x):
            length_x = x[1].shape[0]
            moment_arm_array = np.zeros(length_x)
            transformed_coordinates_vector = np.absolute(x[1]) - 0.5*self.p.breadth
            for i in range(length_x):
                if transformed_coordinates_vector[i]!=0:
                    moment_arm_array[i] = 1/transformed_coordinates_vector[i]**2
            return moment_arm_array
    
        #For torsional spring
        def clamped_neutral_axis(x):          
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1],0.1))
            #return np.isclose(x[0], 0)

        if 2 in self.p['uncertainties']:
            #Linear Spring
            k_x = self.p.k_x
            k_y = self.p.k_y
            self.k_x = df.fem.Constant(self.experiment.mesh, k_x)
            self.k_y = df.fem.Constant(self.experiment.mesh, k_y)
    
            spring_stiffness = ufl.as_matrix([[self.k_x, 0], [0, self.k_y]])
            self.spring_stress = ufl.dot(spring_stiffness,self.u_trial)

            ds = self.experiment.create_neumann_boundary()  
            self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx + ufl.dot(self.spring_stress, self.v) * ds(2)
            self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        elif 3 in self.p['uncertainties']:
            #Torsion Spring
            K_torsion = self.p.K_torsion
            self.K_torsion = df.fem.Constant(self.experiment.mesh, K_torsion)
            moment_arm = df.fem.Function(self.experiment.V_scalar)

            moment_arm.interpolate(moment_arm_)
            spring_stiffness = ufl.as_matrix([[self.K_torsion*moment_arm, 0], [0, 0]])
            self.spring_stress = ufl.dot(spring_stiffness,self.u_trial)

            bc1_facets = df.mesh.locate_entities_boundary(self.experiment.mesh, 0, clamped_neutral_axis)
            bc1_dofs_sub1 = df.fem.locate_dofs_topological(self.experiment.V.sub(1), 0, bc1_facets)
            bc1 = df.fem.dirichletbc(ScalarType(0), bc1_dofs_sub1, self.experiment.V.sub(1))

            bc2_facets = df.mesh.locate_entities_boundary(self.experiment.mesh, 0, clamped_neutral_axis)
            bc2_dofs_sub0 = df.fem.locate_dofs_topological(self.experiment.V.sub(0), 0, bc2_facets)
            bc2 = df.fem.dirichletbc(ScalarType(0), bc2_dofs_sub0, self.experiment.V.sub(0))

            ds = self.experiment.create_neumann_boundary()
            self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx + ufl.dot(self.spring_stress, self.v) * ds(2)
            self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        else:
            self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx
            self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Random E and nu fields.
    def random_field_generator(self, field_function_space, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values, ktol):
        random_field = Randomfield(field_function_space, cov_name, mean, correlation_length1, correlation_length2, variance, no_eigen_values, ktol)
        #random_field.solve_covariance_EVP()
        return random_field

    def parameters_conversion(self, lognormal_mean, lognormal_sigma):
        from math import sqrt
        normal_mean = np.log(lognormal_mean/sqrt(1 + (lognormal_sigma/lognormal_mean)**2))
        normal_sigma = np.log(1 + (lognormal_sigma/lognormal_mean)**2)
        return normal_mean, normal_sigma
    
    def get_lames_constants(self,):
        # Random E and nu fields.
        E_mean, E_variance = self.parameters_conversion(self.p.E, 30e9)#3
        Nu_mean, Nu_variance = self.parameters_conversion(self.p.nu, 0.05)#0.03

        self.E_randomfield  = self.random_field_generator(self.field_function_space,'squared_exp', E_mean, 0.3, 0.05, E_variance, 3, 0.01) 
        self.E_randomfield.create_random_field(_type='random', _dist='LN')

        self.nu_randomfield = self.random_field_generator(self.field_function_space,'squared_exp', Nu_mean, 0.3, 0.05, Nu_variance, 3, 0.01)
        self.nu_randomfield.create_random_field(_type='random', _dist='LN')

        lame1 = (self.E_randomfield.field.vector[:] * self.nu_randomfield.field.vector[:])/((1 + self.nu_randomfield.field.vector[:])*(1-2*self.nu_randomfield.field.vector[:]))
        lame2 = self.E_randomfield.field.vector[:]/(2*(1+self.nu_randomfield.field.vector[:]))
        return lame1, lame2

    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 

    #Probablistic
    #def sigma(self, u):
    #    return self.lambda_() * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu()*self.epsilon(u)

    #Deterministic
    def sigma(self, u):
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu*self.epsilon(u)
    
    def project_fenicsx(self, v, V, dx, u=None):
        projv = ufl.TrialFunction(V)
        v_ = ufl.TestFunction(V)
        a_proj = ufl.inner(projv, v_) * dx
        L_proj = ufl.inner(v, v_) * dx
        if u is None:
            solver = df.fem.petsc.LinearProblem(a_proj, L_proj)
            uh = solver.solve()
            return uh
        else:
            solver = df.fem.petsc.LinearProblem(a_proj, L_proj, u=u)
            solver.solve()

    def solve(self, t=1.0):        
        self.displacement = self.weak_form_problem.solve()
        #self.strain_derivative_reshaped = self.strain_derivative.x.array.reshape((-1,4,2))

        #self.stress = self.sigma(self.displacement)
        
        # TODO make some switch in sensor definition to trigger this...
        #self.compute_residual()

        #Calculation of reaction forces 
        #self.internal_forces = df.fem.assemble_vector(df.fem.form(ufl.inner(self.sigma(self.displacement), self.epsilon(self.v)) * ufl.dx))
        #self.external_forces = df.fem.assemble_vector(df.fem.form(self.L))
        #self.int_forces=df.fem.assemble_vector(df.fem.form(ufl.action(self.a, self.displacement)))

        self.residual = ufl.action(self.a, self.displacement) - self.L
        self.residual_numeric = df.fem.assemble_vector(df.fem.form(self.residual))

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

    def pv_plot(self, t=0):
        # paraview output
        
        # Displacement Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Displacement.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh)
            xdmf.write_function(self.displacement)

        # Strain Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Strain_DG0.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.strain_DG0)
#
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Strain_CG1.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.strain_CG1)

        # Strain Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Strain.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh)
            xdmf.write_function(self.strain)
        
        # Strain derivative Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Displacement_Double_Derivative.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh)
            xdmf.write_function(self.displacement_double_derivative)

        #with df.io.XDMFFile(self.experiment.mesh.comm, "Strain_commonspace.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.strain_commonspace) 

        # Youngs Modulus Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Youngs_Modulus.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    self.E_randomfield.field.name = "Youngs Modulus"
        #    xdmf.write_function(self.E_randomfield.field)
        #    #xdmf.write_function(self.p.nu.field)

        # Poissons ratio Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Poissons_Ratio.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    self.nu_randomfield.field.name = "Poissons Ratio"
        #    xdmf.write_function(self.nu_randomfield.field)