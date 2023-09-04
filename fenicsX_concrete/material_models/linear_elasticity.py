#import sys
#print(sys.path)
import dolfinx as df
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
        #self.dsn = self.experiment.identify_domain_sub_boundaries(self.p.lower_limit, self.p.upper_limit)

        if 0 in self.p['uncertainties'] and self.p.constitutive == 'orthotropic':
            #self.E_m = df.fem.Constant(self.experiment.mesh, self.p.E_m)
            #self.E_d = df.fem.Constant(self.experiment.mesh, self.p.E_d)
            self.E_1 = df.fem.Constant(self.experiment.mesh, self.p.E_1)
            self.E_2 = df.fem.Constant(self.experiment.mesh, self.p.E_2)
            self.nu_12 = df.fem.Constant(self.experiment.mesh, self.p.nu_12)
            self.G_12 = df.fem.Constant(self.experiment.mesh, self.p.G_12)
        
        # Constant E and nu fields.
        if 0 in self.p['uncertainties'] and self.p.constitutive == 'isotropic':
            self.E = df.fem.Constant(self.experiment.mesh, self.p.E)
            self.nu = df.fem.Constant(self.experiment.mesh, self.p.nu)

            #self.lambda_ = df.fem.Constant(self.experiment.mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
            #self.mu = df.fem.Constant(self.experiment.mesh, E / (2.0 * (1.0 + nu)))

        # Random E and nu fields.
        if 1 in self.p['uncertainties']:
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
            self.k_y = df.fem.Constant(self.experiment.mesh, self.p.k_x)
    
            self.spring_stiffness = ufl.as_matrix([[self.k_x, 0], [0, self.k_y]])
            self.spring_stress = ufl.dot(self.spring_stiffness,self.u_trial)

            #self.ds = self.experiment.create_neumann_boundary()  
            self.internal_force_term = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx #- ufl.dot(self.spring_stress, self.v) * self.ds(2)
            
            self.calculate_bilinear_form()

            self.solver = PETSc.KSP().create(self.experiment.mesh.comm)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
            #self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            
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

            #ds = self.experiment.create_neumann_boundary()
            self.internal_force_term = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx #- ufl.dot(self.spring_stress, self.v) * self.ds(2)
            self.spring_force_term = ufl.dot(self.spring_stress, self.v) * self.ds(2)
            self.a = self.internal_force_term - self.spring_force_term
            
            
            self.bilinear_form = df.fem.form(self.a)
            self.solver = PETSc.KSP().create(self.experiment.mesh.comm)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
            #self.weak_form_problem = df.fem.petsc.LinearProblem(self.a, self.L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        else:
            self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx
            

            self.bilinear_form = df.fem.form(self.a)
            self.solver = PETSc.KSP().create(self.experiment.mesh.comm)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
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
        if self.p.problem == 'tensile_test':
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load[0], self.p.load[1]))) #self.p.load
            self.L =  ufl.dot(self.T, self.v) * self.dsn(5)
            #self.ds = self.experiment.create_neumann_boundary()
            #if  self.p.dirichlet_bdy == 'left':
            #    self.L =  ufl.dot(self.T, self.v) * self.ds(1) 
            #elif self.p.dirichlet_bdy == 'bottom':
            #    self.L =  ufl.dot(self.T, self.v) * self.ds(3)

        elif self.p.problem == 'bending_test':
            if self.p.dim == 2:
                #f = df.Constant((0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) #0, -self.p.rho*self.p.g
            elif self.p.dim == 3:
                #f = df.Constant((0, 0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
            else:
                raise Exception(f'wrong dimension {self.p.dim} for problem setup')
              
            self.L =  ufl.dot(f, self.v) * ufl.dx

        elif self.p.problem == 'bending+tensile_test':
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load[0], self.p.load[1]))) #self.p.load
            self.L =  ufl.dot(self.T, self.v) * self.ds(1) 
            if self.p.dim == 2:
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p.rho*self.p.g))) #0, -self.p.rho*self.p.g
            elif self.p.dim == 3:
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p.rho*self.p.g))) 
            else:
                raise Exception(f'wrong dimension {self.p.dim} for problem setup')
            self.L =  ufl.dot(self.T, self.v) * self.ds(1)  + ufl.dot(f, self.v) * ufl.dx

        elif self.p.problem == 'tensile_xy_test':
            self.T1 = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load[0], self.p.load[1]))) #self.p.load
            self.T2 = df.fem.Constant(self.experiment.mesh, ScalarType((self.p.load[1], self.p.load[0])))
            self.L =  ufl.dot(self.T1, self.v) * self.ds(1) + ufl.dot(self.T2, self.v) * self.ds(3) 
        else:
            print('wrong problem type')
            exit()

    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 

    #Probablistic
    #def sigma(self, u):
    #    return self.lambda_() * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu()*self.epsilon(u)

    #Deterministic
    def sigma(self, u):
        if self.p.constitutive == 'orthotropic':
            #ortho3 - same results as ortho2 when E_d = 0
            #denominator = self.E_m + self.E_d - (self.E_m - self.E_d)*self.nu_12**2
            #cmatrix_11 = (self.E_m + self.E_d)*(self.E_m + self.E_d) / denominator
            #cmatrix_22 = (self.E_m - self.E_d)*(self.E_m + self.E_d) / denominator
            #cmatrix_33 = self.G_12
            #cmatrix_12 = self.nu_12*(self.E_m - self.E_d)*(self.E_m + self.E_d) / denominator
#
            ##1st attempt
            #c_matrix_voigt = ufl.as_matrix([[cmatrix_11, cmatrix_12, 0],
            #                     [cmatrix_12, cmatrix_22, 0],
            #                     [0, 0, cmatrix_33]])
#
            #epsilon_tensor = self.epsilon(u)
            #epsilon_voigt = ufl.as_vector([epsilon_tensor[0,0], epsilon_tensor[1,1], 2*epsilon_tensor[0,1]])
            #stress_voigt = ufl.dot(c_matrix_voigt, epsilon_voigt) 
            #stress_tensor = ufl.as_tensor([[stress_voigt[0], stress_voigt[2]], [stress_voigt[2], stress_voigt[1]]])
            #return stress_tensor
        
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
        
        elif self.p.constitutive == 'isotropic':
            #iso1 - gives correct results
            #cmatrix_11 = self.E/ (1 - self.nu**2)
            #cmatrix_22 = self.E/ (1 - self.nu**2)
            #cmatrix_33 = self.E/ (2*(1 + self.nu))
            #cmatrix_12 = self.nu*self.E / (1 - self.nu)

            denominator = 1 - self.nu**2
            cmatrix_11 = self.E/ denominator
            cmatrix_22 = self.E/ denominator
            cmatrix_33 = self.E/ (2*(1 + self.nu))
            cmatrix_12 = self.nu*self.E / denominator

            c_matrix_voigt = ufl.as_matrix([[cmatrix_11, cmatrix_12, 0],
                                 [cmatrix_12, cmatrix_22, 0],
                                 [0, 0, cmatrix_33]])

            epsilon_tensor = self.epsilon(u)
            epsilon_voigt = ufl.as_vector([epsilon_tensor[0,0], epsilon_tensor[1,1], 2*epsilon_tensor[0,1]])
            stress_voigt = ufl.dot(c_matrix_voigt, epsilon_voigt) 
            stress_tensor = ufl.as_tensor([[stress_voigt[0], stress_voigt[2]], [stress_voigt[2], stress_voigt[1]]])
            return stress_tensor
            #return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.mu*self.epsilon(u)
    
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

        if 2 in self.p['uncertainties']:
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

        else:
            # Assemble the bilinear form A   and apply Dirichlet boundary condition to the matrix
            self.A = df.fem.petsc.assemble_matrix(self.bilinear_form, bcs=self.experiment.bcs)
            self.A.assemble()
            self.solver.setOperators(self.A)

            self.linear_form = df.fem.form(self.L)
            self.b = df.fem.petsc.create_vector(self.linear_form)

            # Update the right hand side reusing the initial vector
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            df.fem.petsc.assemble_vector(self.b, self.linear_form)

            # Apply Dirichlet boundary condition to the vector
            df.fem.petsc.apply_lifting(self.b, [self.bilinear_form], [self.experiment.bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            df.fem.petsc.set_bc(self.b, self.experiment.bcs)

            # Solve linear problem
            self.displacement = df.fem.Function(self.experiment.V)
            self.solver.solve(self.b, self.displacement.vector)
            self.displacement.x.scatter_forward()

        #self.displacement = self.weak_form_problem.solve()








        #self.strain_derivative_reshaped = self.strain_derivative.x.array.reshape((-1,4,2))
        #self.stress = self.sigma(self.displacement)
        
        # TODO make some switch in sensor definition to trigger this...
        #self.compute_residual()

        #Calculation of reaction forces 
        #self.internal_forces = df.fem.assemble_vector(df.fem.form(ufl.inner(self.sigma(self.displacement), self.epsilon(self.v)) * ufl.dx))
        #self.external_forces = df.fem.assemble_vector(df.fem.form(self.L))
        #self.int_forces=df.fem.assemble_vector(df.fem.form(ufl.action(self.a, self.displacement)))

        #self.residual = ufl.action(self.a, self.displacement) - self.L
        #self.residual_numeric = df.fem.assemble_vector(df.fem.form(self.residual))

        """ # EUCLID Implementation:
        self.residual = ufl.action(self.a, self.displacement) - self.L
        self.residual_numeric = df.fem.assemble_vector(df.fem.form(self.residual))

        if 0 in self.p['uncertainties'] and 2 not in self.p['uncertainties'] and 3 not in self.p['uncertainties']:
            internal_forces = df.fem.assemble_vector(df.fem.form(ufl.action(self.a, self.displacement)))
            external_forces = df.fem.assemble_vector(df.fem.form(self.L))
        else:
            internal_forces = df.fem.assemble_vector(df.fem.form(ufl.action(self.internal_force_term, self.displacement)))
            spring_forces = df.fem.assemble_vector(df.fem.form(ufl.action(self.spring_force_term, self.displacement)))
            external_forces = df.fem.assemble_vector(df.fem.form(self.L))
            self.force_x_vector = spring_forces.array[self.experiment.bc_x_dof] #self.residual_numval
            self.force_y_vector = spring_forces.array[self.experiment.bc_y_dof] """

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

    """ def solve_inverse_problem(self, displacement_measured, reaction_force_measured, t=1.0):
        self.displacement_measured_function = df.fem.Function(self.experiment.V)
        self.displacement_measured_function.x.array[:] = displacement_measured
        self.euclid_residual = ufl.action(self.a, self.displacement_measured_function) - self.L
        #lhs = df.fem.assemble_vector(df.fem.form(ufl.action(self.a, self.displacement_measured_function)))
        #rhs = df.fem.assemble_vector(df.fem.form(self.L))
        self.euclid_residual_numeric = df.fem.assemble_vector(df.fem.form(self.euclid_residual))
        #self.spring_force_inverse_problem = df.fem.assemble_vector(df.fem.form(ufl.action(self.spring_force_term, self.displacement_measured_function)))

        self.dirichlet_dofs = np.concatenate((self.experiment.bc_x_dof, self.experiment.bc_y_dof))
        self.momentum_balance = np.delete(self.euclid_residual_numeric.array, self.dirichlet_dofs)
        #self.momentum_balance = self.euclid_residual_numeric.array #For the case of springs, where LHS nodes becomes the internal nodes.

        self.force_balance_x = np.sum(self.euclid_residual_numeric.array[self.experiment.bc_x_dof]) - reaction_force_measured[0]
        self.force_balance_y = np.sum(self.euclid_residual_numeric.array[self.experiment.bc_y_dof]) - reaction_force_measured[1]
        self.force_balance = np.array([self.force_balance_x, self.force_balance_y])
        self.reaction_force_model = np.array([np.sum(self.euclid_residual_numeric.array[self.experiment.bc_x_dof]), np.sum(self.euclid_residual_numeric.array[self.experiment.bc_y_dof]) ]) """

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