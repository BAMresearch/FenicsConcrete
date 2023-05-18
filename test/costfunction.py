import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import matplotlib.pyplot as matplot
import math

#########################################################################
#########################################################################
#1st Step - Data Generation
#########################################################################
#########################################################################

p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['problem'] =  'tensile_test'    #'tensile_test' #'bending_test' 
p['degree'] = 1
p['num_elements_length'] = 50
p['num_elements_breadth'] = 10
p['dim'] = 2
# Uncertainty type:
# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs.
# 3: Torsion Springs
p['uncertainties'] = [0]

p['constitutive'] = 'isotropic'
p['nu'] = 0.28 

# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3,0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
problem.solve() 
displacement_data = problem.displacement.x.array
disp = np.copy(displacement_data)


# Kgmms⁻2/mm², mm, kg, sec, N
p['constitutive'] = 'orthotropic'
p['uncertainties'] = [0,2]
p['E_m'] = 210e6
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  210e6/(2*(1+0.28)) #(0.5*1e5)/(1+0.3)
p['k_x'] = 1e12
p['k_y'] = 1e12
experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

scaler = 500e6
def forward_model_run(parameters):
    # Function to run the forward model
    #problem.E.value = parameters[0]
    #problem.nu.value = parameters[1]
    problem.E_m.value = parameters[0]*scaler
    problem.E_d.value = parameters[1]*scaler
    problem.solve()
    return problem.displacement.x.array

from numpy import linalg as LA
def cost_function(param):
    # Function to calculate the cost function
    displacement_model = forward_model_run(param)  
    delta_displacement = displacement_model - disp
    #print('Optimisation Parameters',param[0], param[1])
    return   np.dot(delta_displacement, delta_displacement) #+ LA.norm(param)

import matplotlib.pyplot as plt
from matplotlib import cm

def cost_function_plot():
    
    #counter=0
    #E_values = np.linspace(185e9,225e9,30)
    #E_values = np.linspace(100e6,400e6,30)
    #nu_values = np.linspace(0.01,0.45,15)
    #E_buildup, nu_buildup = np.meshgrid(E_values, nu_values)
    E_m_values = np.linspace(0.3,0.5,30)
    E_d_values = np.linspace(0.,0.1,15)
    E_m_buildup, E_d_buildup = np.meshgrid(E_m_values, E_d_values)
    counter=0
    cost_func_val = np.zeros((E_m_buildup.shape[0],E_d_buildup.shape[1]))
    for i in range(E_m_buildup.shape[0]):
        for j in range(E_d_buildup.shape[1]):
            cost_func_val[i,j] = cost_function(np.array([E_m_buildup[i,j],E_d_buildup[i,j]]))
            counter+=1
            #print(cost_func_val[i,j])
        print(counter)

    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Surface(z=np.log10(cost_func_val), x= E_m_values, y=E_d_values)])
    fig.update_layout(title='Cost Function Vs. Parameters',autosize=False,width=950, height=950,)
    #fig.update_layout(scene=dict(zaxis=dict(dtick=1, type='log')))
    fig.show()
    """     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(E_m_buildup, E_d_buildup, cost_func_val,  cmap=cm.coolwarm, edgecolor = 'black',
                           linewidth=0, antialiased=False)
 
    # Add a color bar which maps values to colors.
    ax.set_xlabel('E_m')
    ax.set_ylabel('E_d')
    ax.set_zlabel('Cost function value')
    ax.zaxis._set_scale('log')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show() """

cost_function_plot()

