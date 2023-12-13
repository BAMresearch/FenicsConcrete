#import sys
#print(sys.path)
import dolfinx as df
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, apply_lifting, set_bc, LinearProblem
import ufl
from petsc4py.PETSc import ScalarType
from fenicsX_concrete.randomfieldModified import Randomfield
from fenicsX_concrete.material_models.material import MaterialProblem
from fenicsX_concrete.helpers import Parameters
import numpy as np
from petsc4py import PETSc

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

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        default_p = Parameters()

        self.p = default_p + self.p
        self.ds = self.experiment.identify_domain_boundaries() # Domain's boundary
        self.ds_sub = self.experiment.identify_domain_sub_boundaries(self.p.lower_limit, self.p.upper_limit)
  
        # Constant E and nu fields.
        if 0 in self.p['uncertainties'] and self.p.constitutive == 'isotropic':
            self.E = df.fem.Constant(self.experiment.mesh, self.p.E)
            self.nu = df.fem.Constant(self.experiment.mesh, self.p.nu)

            self.lambda_ = df.fem.Constant(self.experiment.mesh, self.p.E * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu)))
            self.mu = df.fem.Constant(self.experiment.mesh, self.p.E / (2.0 * (1.0 + self.p.nu)))

        elif 0 in self.p['uncertainties'] and self.p.constitutive == 'orthotropic':
            self.E_1 = df.fem.Constant(self.experiment.mesh, self.p.E_1)
            self.E_2 = df.fem.Constant(self.experiment.mesh, self.p.E_2)
            self.nu_12 = df.fem.Constant(self.experiment.mesh, self.p.nu_12)
            self.G_12 = df.fem.Constant(self.experiment.mesh, self.p.G_12)

        # Random E and nu fields.
        elif 1 in self.p['uncertainties']:
            self.field_function_space = df.fem.FunctionSpace(self.experiment.mesh, ("CG", 1))
            self.lambda_ = df.fem.Function(self.field_function_space)
            self.mu = df.fem.Function(self.field_function_space)

            lame1, lame2 = self.get_lames_constants()
            self.lambda_.vector[:] = lame1
            self.mu.vector[:] = lame2  #make this vector as a fenics constant array. Update the lame1 and lame2 in each iteration.
        
        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.experiment.V)
        self.v = ufl.TestFunction(self.experiment.V)
        
        self.apply_neumann_bc()

        if self.p.body_force == True:
            if self.p.dim == 2:
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) #0, -self.p.rho*self.p.g
                self.L +=  ufl.dot(f, self.v) * ufl.dx
            elif self.p.dim == 3:
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
                self.L +=  ufl.dot(f, self.v) * ufl.dx
            else:
                raise Exception(f'wrong dimension {self.p.dim} for problem setup') 

        #For torsional spring
        def moment_arm_(x):
            length_x = x[1].shape[0]
            moment_arm_array = np.zeros(length_x)
            transformed_coordinates_vector = np.absolute(x[1]) - 0.5*self.p.breadth
            for i in range(length_x):
                if transformed_coordinates_vector[i]!=0:   #### must be corrected
                    moment_arm_array[i] = 1/transformed_coordinates_vector[i]**2
            return moment_arm_array
    
        #For torsional spring
        def clamped_neutral_axis(x):          
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1],0.5*self.p.breadth))
            #return np.isclose(x[0], 0)

        if 2 in self.p['uncertainties']:
            #Linear Spring
            self.k_x = df.fem.Constant(self.experiment.mesh, self.p.k_x)
            self.k_y = df.fem.Constant(self.experiment.mesh, self.p.k_y)
    
            self.spring_stiffness = ufl.as_matrix([[self.k_x, 0], [0, self.k_y]])
            self.spring_stress = ufl.dot(self.spring_stiffness,self.u_trial)

            #self.ds = self.experiment.create_neumann_boundary()  
            self.internal_force_term = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx #- ufl.dot(self.spring_stress, self.v) * self.ds(2)
            
            self.calculate_bilinear_form()

            self.solver = PETSc.KSP().create(self.experiment.mesh.comm)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
            #self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        else:
            self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx 
            #self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

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
    
    def calculate_bilinear_form(self):
        if 2 in self.p['uncertainties']:
            if self.p.dirichlet_bdy == 'left':
                self.spring_force_term = ufl.dot(self.spring_stress, self.v) * self.ds(2)
            elif self.p.dirichlet_bdy == 'bottom':
                self.spring_force_term = ufl.dot(self.spring_stress, self.v) * self.ds(4)
            self.a = self.internal_force_term - self.spring_force_term
            self.bilinear_form = df.fem.form(self.a)
    
    def apply_neumann_bc(self):
        # Selects the problem which you want to solve
        self.T = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load[0], self.p.load[1]))) #self.p.load
        self.L =  ufl.dot(self.T, self.v) * self.ds_sub(5)

             
    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 

    #Deterministic
    def sigma(self, u):
        if self.p.constitutive == 'isotropic':
            #self.delta_theta = df.fem.Function(self.experiment.V_scalar) #self.V.mesh.geometry.dim
            #self.delta_theta.interpolate(lambda x: 2.0*x[0])
            #self.delta_theta = df.fem.Constant(self.experiment.mesh, 5.0)
            #self.beta = 0.2
            #return stress_tensor #+ ufl.Identity(len(u))*self.delta_theta*self.beta
            return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*self.mu*self.epsilon(u) #+ ufl.Identity(len(u))*self.delta_theta*self.beta

        elif self.p.constitutive == 'orthotropic':    
            denominator = self.E_1 - self.E_2*self.nu_12**2
            cmatrix_11 = self.E_1**2 / denominator
            cmatrix_22 = (self.E_1*self.E_2)/ denominator
            cmatrix_33 = self.G_12
            cmatrix_12 = self.nu_12*self.E_1*self.E_2 / denominator

            c_matrix_voigt = ufl.as_matrix([[cmatrix_11, cmatrix_12, 0],
                                 [cmatrix_12, cmatrix_22, 0],
                                 [0, 0, cmatrix_33]])

            epsilon_tensor = self.epsilon(u)
            epsilon_voigt = ufl.as_vector([epsilon_tensor[0,0], epsilon_tensor[1,1], 2*epsilon_tensor[0,1]])
            stress_voigt = ufl.dot(c_matrix_voigt, epsilon_voigt) 
            stress_tensor = ufl.as_tensor([[stress_voigt[0], stress_voigt[2]], [stress_voigt[2], stress_voigt[1]]])
            return stress_tensor
        
    def solve(self, t=1.0):  
        if 0 in self.p['uncertainties']:
            problem = LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            self.displacement = problem.solve()

        elif 2 in self.p['uncertainties']:
            # Assemble the bilinear form A   and apply Dirichlet boundary condition to the matrix
            self.A = df.fem.petsc.assemble_matrix(self.bilinear_form, bcs=[] ) 
            self.A.assemble()
            self.solver.setOperators(self.A)

            self.linear_form = df.fem.form(self.L)
            self.b = df.fem.petsc.create_vector(self.linear_form)

            # Update the right hand side reusing the initial vector
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            df.fem.petsc.assemble_vector(self.b, self.linear_form)

            # Apply Dirichlet boundary condition to the vector
            df.fem.petsc.apply_lifting(self.b, [self.bilinear_form], [[]])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            df.fem.petsc.set_bc(self.b, [])

            # Solve linear problem
            self.displacement = df.fem.Function(self.experiment.V)
            self.solver.solve(self.b, self.displacement.vector)
            self.displacement.x.scatter_forward()

        #self.displacement = self.weak_form_problem.solve()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

    def pv_plot(self, name, t=0):
        # paraview output
        
        # Displacement Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, name, "w") as xdmf:
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
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Strain.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.strain)
        #
        ## Strain derivative Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Displacement_Double_Derivative.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.displacement_double_derivative)

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