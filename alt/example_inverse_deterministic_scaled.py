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
p['problem'] =  'tensile_test'     #'bending_test' 
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

# N/m², m, kg, sec, N
#p['length'] = 5
#p['breadth'] = 1
#p['load'] = [1e6,0] #[0, -10] #
#p['rho'] = 7750
#p['g'] = 9.81
#p['E'] = 210e9 


# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [0,1e3] 
p['dirichlet_bdy'] = 'bottom'
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa


experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
#problem.add_sensor(fenicsX_concrete.sensors.ReactionForceSensor())
problem.solve() 
#reaction_force_data = problem.sensors['ReactionForceSensor'].data[-1]
displacement_data = problem.displacement.x.array
problem.pv_plot("Displacement_cantilever.xdmf")

#########################################################################
#########################################################################
#2nd Step - Inverse Problem
#########################################################################
#########################################################################

p['uncertainties'] = [0,2]
p['constitutive'] = 'orthotropic'

# Kgmms⁻2/mm², mm, kg, sec, N
p['E_m'] = 180e6 #1e5
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  p['E_m']/(2*(1+p['nu_12'])) #(0.5*1e5)/(1+0.3)
p['k_x'] = 1e8
p['k_y'] = 1e8

""" # N/m², m, kg, sec, N
p['E_m'] = 180e9 #1e5
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  p['E_m']/(2*(1+p['nu_12'])) #(0.5*1e5)/(1+0.3)
p['k_x'] = 1e13
p['k_y'] = 1e12 """

experiment = fenicsX_concrete.concreteSlabExperiment(p)  
problem = fenicsX_concrete.LinearElasticity(experiment, p)


E_scaler = 500e6
G_12_scaler = 250e6
def forward_model_run(parameters):
    # Function to run the forward model
    problem.E_m.value = parameters[0]*E_scaler
    problem.E_d.value = parameters[1]*E_scaler
    problem.nu_12.value = 0.28#parameters[2]#parameters[2]
    problem.G_12.value = 210e6/(2*(1+0.28))# + problem.E_m.value/(2*(1+problem.nu_12.value))
    problem.k_x.value = 1e12            #parameters[4]               #abs(1/parameters[0] -1) +1e9 #*p['E_m']
    problem.k_y.value = 1e12            #parameters[5]               #abs(1/parameters[1] -1) +1e9#*p['E_m']
    problem.solve()
    return problem.displacement.x.array


from numpy import linalg as LA
total_model_error = []
displacement_model_error = []

sparsity_factor = 1e-1
def cost_function(param):
    # Function to calculate the cost function
    displacement_model = forward_model_run(param)  
    delta_displacement = displacement_model - displacement_data
    print('Inferred Parameters',param[0], param[1])
    #print('Cost Function', np.dot(delta_displacement, delta_displacement), sparsity_factor*LA.norm(param, ord=1))
    function_evaluation = np.dot(delta_displacement, delta_displacement) 
    cost_function_value = function_evaluation +  sparsity_factor*LA.norm(param, ord=1)
    displacement_model_error.append(function_evaluation)
    total_model_error.append(cost_function_value)
    return cost_function_value

from scipy.optimize import minimize, least_squares, LinearConstraint

constraint_matrix = np.array([[1,-1]])
constraint = LinearConstraint(constraint_matrix, [0])
start_point = np.array([0.9, 0.6]) #, 1e8, 1e8 #[0.9, 0.6]
parameter_bounds = [(0, np.inf), (0, np.inf)] #, (0, np.inf), (0, np.inf) L-BFGS-B
#res = minimize(cost_function, start_point, method='Powell', bounds=parameter_bounds,#0.50.5
#              options={ 'ftol': 1e-40, 'disp': True, 'maxiter':400}) #'ftol': 1e-10, 
res = minimize(cost_function, start_point, method='trust-constr', bounds=parameter_bounds,constraints=[constraint],
              options={'disp': True, 'maxiter':400}) #'ftol': 1e-10, 
print(res.x) 
print("Inferred Values",np.multiply(res.x, np.array([E_scaler, E_scaler])))
print(p['G_12'])


""" import plotly.express as px
fig = px.line(x=total_model_error, y=cost_function_values[45:], markers=True, title='Cost Function Curve', log_y=True)
fig.update_layout(
    title_text='Cost Function Curve',
)
fig.show()

import matplotlib.pyplot as plt
from matplotlib import cm

E_m_trials = np.array(parameter_values['E_m'])
E_d_trials = np.array(parameter_values['E_d'])
cost_function_trials = np.array(cost_function_values) """


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.range(1,len(total_model_error)+1), y=total_model_error,
                    mode='lines',
                    name='lines'))
fig.add_trace(go.Scatter(x=np.range(1,len(total_model_error)+1), y=displacement_model_error,
                    mode='lines+markers',
                    name='lines+markers'))
fig.show()





































"""
#Isotropic data, istropic identification of tensile test
start_point = np.array([50e9, 0.15]) #, 1e8, 1e8
parameter_bounds = [(0, np.inf), (0, 0.45)] #, (0, np.inf), (0, np.inf)
res = minimize(cost_function, start_point, method='Nelder-Mead', bounds=parameter_bounds,#0.50.5
              options={'disp': True, 'maxiter':400}) 
print(res.x) 
print(p['G_12']) 

#parameter_values = {'E_m':[], 'E_d':[]}
#parameter_values['E_m'].append(param[0])
#parameter_values['E_d'].append(param[1])

"""


