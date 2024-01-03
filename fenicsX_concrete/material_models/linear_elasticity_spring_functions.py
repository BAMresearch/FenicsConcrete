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
from slepc4py import SLEPc
from scipy.optimize import root
import math

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

        #Variable definitions for damage modelling
        self.V_scalar = df.fem.functionspace(self.experiment.mesh, ("Lagrange", 1)) 
        self.damage_locations = df.fem.Constant(self.experiment.mesh, self.p.damage_locations)

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
        
        weight_load = df.fem.Constant(self.experiment.mesh, ScalarType(self.p.weight)) 
        self.M =  self.p.rho*ufl.dot(self.v, self.u_trial) * ufl.dx

        self.apply_neumann_bc()

        if self.p.body_force == True:
            if self.p.dim == 2:
                f = df.fem.Constant(self.experiment.mesh, ScalarType(self.p.weight)) #0, -self.p.rho*self.p.g
                self.L +=  ufl.dot(f, self.v) * ufl.dx
            elif self.p.dim == 3:
                f = df.fem.Constant(self.experiment.mesh, ScalarType(self.p.weight)) 
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

        lame1 = (self.E_randomfield.field.vector.array * self.nu_randomfield.field.vector.array)/((1 + self.nu_randomfield.field.vector.array)*(1-2*self.nu_randomfield.field.vector.array))
        lame2 = self.E_randomfield.field.vector.array/(2*(1+self.nu_randomfield.field.vector.array))

        #lame1 = (self.E_randomfield.field.vector[:] * self.nu_randomfield.field.vector[:])/((1 + self.nu_randomfield.field.vector[:])*(1-2*self.nu_randomfield.field.vector[:]))
        #lame2 = self.E_randomfield.field.vector[:]/(2*(1+self.nu_randomfield.field.vector[:]))
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
        if self.p.dim == 2:
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType(self.p.load)) #self.p.load
        elif self.p.dim == 3:
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType(self.p.load)) #self.p.load
        self.L =  ufl.dot(self.T, self.v) * self.experiment.ds(1)

    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 
    
    def damage_coordinate(self, z_coordinate):

        def damage_basis_function(x,t=z_coordinate):
            size_x = x.shape[1]
            damage_field = np.zeros(size_x)
            damage_affected_region = 0.05
            k = -(damage_affected_region - t)**2/math.log(0.0001)
            damage_region_quadrature_pts = np.where((x[2,:] < t + damage_affected_region) & (x[2,:] > t - damage_affected_region))
            damage_field[damage_region_quadrature_pts] =   np.exp(-(x[2,[damage_region_quadrature_pts]]-t)**2/k) 
            return damage_field
     
        return damage_basis_function

    def add_damage(self,):
        #damage_locations = [0.3 ,0.7]
        damage_basis_functions = []

        #xdmf = df.io.XDMFFile(self.experiment.mesh.comm, "damage_distribution.xdmf", "w")
        #xdmf.write_mesh(self.experiment.mesh)
        for counter, value in enumerate(self.damage_locations.value):
            damage_basis_functions.append(df.fem.Function(self.V_scalar))
            damage_basis_functions[counter].interpolate(self.damage_coordinate(value)) #interpolates the damage basis function over the domain
            #xdmf.write_function(damage_basis_functions[counter], counter)
        #xdmf.close()
        self.omega = sum(damage_basis_functions)

    #Deterministic
    def sigma(self, u):
        if self.p.constitutive == 'isotropic':
            #self.delta_theta = df.fem.Function(self.experiment.V_scalar) #self.V.mesh.geometry.dim
            #self.delta_theta.interpolate(lambda x: 2.0*x[0])
            #self.delta_theta = df.fem.Constant(self.experiment.mesh, 5.0)
            #self.beta = 0.2
            #return stress_tensor #+ ufl.Identity(len(u))*self.delta_theta*self.beta
            #function_space_scalar = df.fem.functionspace(self.experiment.mesh, ("Lagrange", self.p.degree, (1,)))
            
            unity = df.fem.Constant(self.experiment.mesh, 1.0) #(unity - omega) * 
            return  (unity-self.omega)*(self.lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * self.mu*self.epsilon(u)) #+ ufl.Identity(len(u))*self.delta_theta*self.beta

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
        problem = LinearProblem(self.a, self.L, bcs=self.experiment.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.displacement = problem.solve()

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

        #self.displacement = self.weak_form_problem.solve()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)


    def solve_eigenvalue_problem(self,):
        self.add_damage()
        # Create eigensolver
        self.stiffness_matrix = assemble_matrix(df.fem.form(self.a), bcs=self.experiment.bcs, diagonal=1)
        self.stiffness_matrix.assemble()
        self.mass_matrix = assemble_matrix(df.fem.form(self.M), bcs=self.experiment.bcs, diagonal=1) #diagonal=1/62831
        self.mass_matrix.assemble()

        # Create eigensolver
        self.eigensolver = SLEPc.EPS().create(comm=self.experiment.mesh.comm)
        self.eigensolver.setOperators(self.stiffness_matrix, self.mass_matrix)
        self.eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        #tol = 1e-9
        #eigensolver.setTolerances(tol=tol)

        #eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        st = self.eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.)

        #st = SLEPc.ST().create(self.experiment.mesh.comm)
        #st.setType(SLEPc.ST.Type.SINVERT)
        #st.setShift(0.1)
        #st.setFromOptions()
        #eigensolver.setST(st)
        #eigensolver.setOperators(K, M)
        #eigensolver.setFromOptions()
        #eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        #eigensolver.setshift(0.0)

        self.eigensolver.setDimensions(nev=65)
        self.eigensolver.solve()
        self.eigensolver.view()
        self.eigensolver.errorView()
        #print(self.eigensolver.getConverged())
        
        #eigensolver.view()
        #eigensolver.errorView()

        #vals = [(i, np.sqrt(eigensolver.getEigenvalue(i))) for i in range(eigensolver.getConverged())]
        #vals.sort(key=lambda x: x[1].real)


    def pv_eigenvalue_plot(self, name, t=0):
        xdmf = df.io.XDMFFile(self.experiment.mesh.comm, name, "w")
        xdmf.write_mesh(self.experiment.mesh)
        eig_v = []
        ef = 0 
        evs = self.eigensolver.getConverged()
        print("Number of converged eigenpairs %d" % evs)
        ur = df.fem.Function(self.experiment.V)
        vr, vi = self.stiffness_matrix.createVecs()

        falpha = lambda x: math.cos(x)*math.cosh(x)+1
        alpha = lambda n: root(falpha, (2*n+1)*math.pi/2.)['x'][0]

        if evs > 0:

            for i in range(evs): #evs          
                eigen_value =self.eigensolver.getEigenpair(i, vr, vi)
                
                if (~np.isclose(eigen_value.real, 1.0)):
                    #Calculation of eigenfrequency from real part of eigenvalue
                    freq_3D = np.sqrt(eigen_value.real)/2/np.pi

                    # Beam eigenfrequency
                    if ef % 2 == 0:
                        # exact solution should correspond to weak axis bending
                        I_bend = self.p.dim_x*self.p.dim_y**3/12.
                    else:
                        # exact solution should correspond to strong axis bending
                        I_bend = self.p.dim_y*self.p.dim_x**3/12.

                    freq_beam = alpha(ef/2)**2*np.sqrt(self.p.E*I_bend/(self.p.rho*self.p.dim_x*self.p.dim_y*self.p.dim_z**4))/2/np.pi

                    print(
                        "Solid FE: {0:8.5f} [Hz] "
                        "Beam theory: {1:8.5f} [Hz]".format(freq_3D, freq_beam))

                    #ur = df.fem.Function(self.experiment.V)
                    ur.vector.array[:] = vr
                    xdmf.write_function(ur,freq_3D) #
                    #xdmf.write_function(ur.copy())
                    #eig_v.append(vr.copy())
                    ef += 1
        xdmf.close()   

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

        # Strain Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Strain.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh)
        #    xdmf.write_function(self.strain)

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