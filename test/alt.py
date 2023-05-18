# Reaction force calculation for a 2D problem

""" self.bc_x= df.fem.dirichletbc(df.fem.Constant(domain=self.mesh, c=1.0), df.fem.locate_dofs_topological(V.sub(0), self.mesh.topology.dim-1, boundary_facets), V.sub(0))
self.bc_y = df.fem.dirichletbc(df.fem.Constant(domain=self.mesh, c=1.0), df.fem.locate_dofs_topological(V.sub(1), self.mesh.topology.dim-1, boundary_facets), V.sub(1))

reaction_force_vector = []
v_reac = df.fem.Function(self.experiment.V)

df.fem.set_bc(v_reac.vector, [self.experiment.bc_x])
computed_force_x = df.fem.assemble_scalar(df.fem.form(ufl.action(self.residual, v_reac)))
reaction_force_vector.append(computed_force_x)
              
df.fem.set_bc(v_reac.vector, [self.experiment.bc_y])
computed_force_y = df.fem.assemble_scalar(df.fem.form(ufl.action(self.residual, v_reac)))
reaction_force_vector.append(computed_force_y)  """   


# Double derivative calculation of displacement fields.

#ufl.Dx(self.experiment.V)
#x = ufl.SpatialCoordinate(self.experiment.mesh)
#d1 = ufl.exp(ufl.Dx(self.displacement.x, 0))
#W = df.fem.FunctionSpace(self.experiment.mesh, ("CG", 1))
#dxUx = df.fem.Expression(d1, W.element.interpolation_points()) #self.experiment.V.element.interpolation_points()
#strain_xx = df.fem.Function(W)
#
##dxUx = df.fem.Expression(d1, self.experiment.V.element.interpolation_points()) #self.experiment.V.element.interpolation_points()
##strain_xx = df.fem.Function(self.experiment.V)
#
#strain_xx.interpolate(dxUx)

# Attempt 2:
#self.strain_space_DG0 = df.fem.TensorFunctionSpace(self.experiment.mesh, ("DG", self.p.degree-1))
#strain_calculation_DG0 = self.epsilon(self.displacement)
#self.strain = self.project_fenicsx(strain_calculation_DG0, self.strain_space_DG0, ufl.dx(self.experiment.mesh)

#Attempt 3:
""" 
self.strain_space = df.fem.TensorFunctionSpace(self.experiment.mesh, ("DG", self.p.degree-1))
#gradient calculated at the interpolation points/quadrature points
strain_expression = df.fem.Expression(self.epsilon(self.displacement), self.strain_space.element.interpolation_points())
#strain_expression = df.fem.Expression(ufl.grad(self.displacement), self.strain_space.element.interpolation_points())
self.strain = df.fem.Function(self.strain_space)
#gradient interpolated at the dof points
self.strain.interpolate(strain_expression)
#self.strain_reshaped = self.strain.x.array.reshape((-1,2,2)
self.displacement_double_derivative_space = df.fem.TensorFunctionSpace(self.experiment.mesh, ("DG", self.p.degree-2), shape=(4,2))
displacement_double_derivative_expression = df.fem.Expression(ufl.grad(ufl.grad(self.displacement)), self.displacement_double_derivative_space.element.interpolation_points())
self.displacement_double_derivative = df.fem.Function(self.displacement_double_derivative_space)
self.displacement_double_derivative.interpolate(displacement_double_derivative_expression) """



            #ortho1 - numerical instability 
            #denominator = self.E_m + self.E_d - (self.E_m - self.E_d)*self.nu_12**2
            #cmatrix_11 = (self.E_m + self.E_d)**2 / denominator
            #cmatrix_22 = self.E_m**2 - self.E_d**2 / denominator
            #cmatrix_33 = self.G_12
            #cmatrix_12 = self.nu_12*(self.E_m**2 - self.E_d**2) / denominator
            
            #ortho2 by removing E_d gives correct results.
            #denominator = self.E_m - self.E_m*self.nu_12**2
            #cmatrix_11 = self.E_m**2 / denominator
            #cmatrix_22 = self.E_m**2 / denominator
            #cmatrix_33 = self.G_12 
            #cmatrix_12 = self.nu_12*(self.E_m**2) / denominator

            #   iso1 - gives correct results
            #   cmatrix_11 = self.E_m/ (1 - self.nu_12**2)
            #   cmatrix_22 = self.E_m/ (1 - self.nu_12**2)
            #   cmatrix_33 = self.E_m/ (2*(1 + self.nu_12))
            #   cmatrix_12 = self.nu_12*self.E_m / (1 - self.nu_12**2)

            #   iso2 - gives correct results same as iso1
            #   cmatrix_33 = 0.5*(1 - self.nu_12)
            #   denominator = self.E_m /(1 - self.nu_12**2)
            #   c_matrix_voigt = denominator*ufl.as_matrix([[1, self.nu_12, 0],
            #            [self.nu_12, 1, 0],
            #            [0, 0, cmatrix_33]])

        # Trial 2
        #if 0 in self.p['uncertainties'] and 2 not in self.p['uncertainties'] and 3 not in self.p['uncertainties']:
        #    internal_forces = df.fem.assemble_vector(df.fem.form(ufl.action(self.a, self.displacement)))
        #    external_forces = df.fem.assemble_vector(df.fem.form(self.L))
        #    self.residual = ufl.action(self.a, self.displacement) - self.L
        #    self.residual_numeric = df.fem.assemble_vector(df.fem.form(self.residual))
        #    #self.residual_numeric = internal_forces - external_forces
        #    print("pause")
        #else:
        #    #self.residual = ufl.action(self.a, self.displacement) - self.L
        #    #self.residual_numeric = df.fem.assemble_vector(df.fem.form(self.residual))
#
        #    internal_forces = df.fem.assemble_vector(df.fem.form(ufl.action(self.a, self.displacement)))
        #    spring_forces = df.fem.assemble_vector(df.fem.form(ufl.action(self.a2, self.displacement)))
        #    external_forces = df.fem.assemble_vector(df.fem.form(self.L))
        #    
        #    self.residual_numval = internal_forces.array[:] - spring_forces.array[:] - external_forces.array[:]
        #    self.force_x_vector = spring_forces.array[self.experiment.bc_x_dof] #self.residual_numval
        #    self.force_y_vector = spring_forces.array[self.experiment.bc_y_dof]
#
        #    self.residual = ufl.action(self.a, self.displacement) - ufl.action(self.a2, self.displacement) - self.L
        #    self.residual_numeric = df.fem.assemble_vector(df.fem.form(self.residual))
        #    self.residual_numeric = internal_forces - spring_forces - external_forces
        #    print("pause")

import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
z = z_data.values
sh_0, sh_1 = z.shape
x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()