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
p['problem'] =  'tensile_test'    #'tensile_test' #'bending_test' #bending+tensile_test
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
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa

p['dirichlet_bdy'] = 'left'
experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

def run_test(exp, prob, dirichlet_bdy, load):
    prob.p.dirichlet_bdy = dirichlet_bdy
    exp.p.dirichlet_bdy = dirichlet_bdy
    prob.p.load = load
    prob.experiment.bcs = prob.experiment.create_displ_bcs(prob.experiment.V)
    prob.apply_neumann_bc()
    prob.calculate_bilinear_form()
    prob.solve()
    #prob.pv_plot("Displacement.xdmf")
    return prob.displacement.x.array

testx_disp = np.copy(run_test(experiment, problem, 'left', [1e3, 0]))
testy_disp = np.copy(run_test(experiment, problem, 'bottom', [0,1e3]))
#tests1_disp = np.copy(run_test(experiment, problem, 'bottom', [1e3,0]))

def combine_test_results(test_results):
    if len(test_results) == 1:
        return test_results[0]
    else:
        return np.concatenate((test_results[0], combine_test_results(test_results[1:])))

list_of_disp = [testx_disp, testy_disp] #, testy_disp
num_of_tests = str(len(list_of_disp)) + ' tests'
displacement_data = combine_test_results(list_of_disp)  

#########################################################################
#########################################################################
#2nd Step - Inverse Problem
#########################################################################
#########################################################################

# Kgmms⁻2/mm², mm, kg, sec, N
p['constitutive'] = 'orthotropic'
p['uncertainties'] = [0,2]
p['E_m'] = 210e6
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  210e6/(2*(1+0.28)) #(0.5*1e5)/(1+0.3)
p['k_x'] = 1e12
p['k_y'] = 1e12

scaler = 500e6
experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.


E_scaler = 500e6
G_12_scaler = 250e6

def forward_model_run(parameters):
    # Function to run the forward model

    problem.E_m.value = parameters[0]*scaler
    problem.E_d.value = parameters[1]*scaler
    problem.nu_12.value = parameters[2]
    problem.G_12.value = parameters[3]*G_12_scaler #problem.E_m.value/(2*(1+problem.nu_12.value))*G_12_scaler
    trialx_disp = np.copy(run_test(experiment, problem, 'left', [1e3, 0]))
    trialy_disp = np.copy(run_test(experiment, problem, 'bottom', [0, 1e3]))
    #trials1_disp = np.copy(run_test(experiment, problem, 'bottom', [1e3, 0]))
    return combine_test_results([trialx_disp, trialy_disp]) #, trialy_disp


from numpy import linalg as LA
cost_function_values = []
parameter_values = {'E_m':[], 'E_d':[]}
sparsity_factor = 1e-1
def cost_function(param):
    # Function to calculate the cost function
    displacement_model = forward_model_run(param)  
    delta_displacement = displacement_model - displacement_data
    print('Inferred Parameters',param)
    #print('Cost Function', np.dot(delta_displacement, delta_displacement), sparsity_factor*LA.norm(param, ord=1))
    function_evaluation = np.dot(delta_displacement, delta_displacement) 
    cost_function_values.append(function_evaluation)
    parameter_values['E_m'].append(param[0])
    parameter_values['E_d'].append(param[1])
    return function_evaluation  #+  sparsity_factor*LA.norm(param, ord=1)

from scipy.optimize import minimize, least_squares, LinearConstraint

constraint_matrix = np.array([[1,-1, 0, 0]]) # 0, 0
constraint = LinearConstraint(constraint_matrix, lb = [0])
start_point = np.array([0.9, 0.6, 0.32, 0.2 ]) #, 1e8, 1e8 #[0.9, 0.6] 0.1, 0.2
parameter_bounds = [(0, 1), (0, 1), (0, 0.45), (0, np.inf) ] #, (0, np.inf), (0, np.inf) L-BFGS-B (0, 0.45), (0, np.inf)
#res = minimize(cost_function, start_point, method='Powell', bounds=parameter_bounds,#0.50.5
#              options={ 'ftol': 1e-40, 'disp': True, 'maxiter':400}) #'ftol': 1e-10, 
res = minimize(cost_function, start_point, method='trust-constr', bounds=parameter_bounds,constraints=[constraint],
              options={'disp': True, 'maxiter':400}) #'ftol': 1e-10, 
print(res.x) 
print("Inferred Values",np.multiply(res.x, np.array([E_scaler, E_scaler,  1, G_12_scaler]))) #1, G_12_scaler
print(p['G_12'])


import plotly.express as px
fig = px.line(x=[i for i in range(46,len(cost_function_values)+1)], y=cost_function_values[45:], markers=True, title='Cost Function Curve', log_y=True)
fig.update_layout(
    title_text='Cost Function Curve',
)
fig.show()

import matplotlib.pyplot as plt
from matplotlib import cm

E_m_trials = np.array(parameter_values['E_m'])
E_d_trials = np.array(parameter_values['E_d'])
cost_function_trials = np.array(cost_function_values)










































"""
#Isotropic data, istropic identification of tensile test
start_point = np.array([50e9, 0.15]) #, 1e8, 1e8
parameter_bounds = [(0, np.inf), (0, 0.45)] #, (0, np.inf), (0, np.inf)
res = minimize(cost_function, start_point, method='Nelder-Mead', bounds=parameter_bounds,#0.50.5
              options={'disp': True, 'maxiter':400}) 
print(res.x) 
print(p['G_12']) """


