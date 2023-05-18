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
#########################################################################
#1st Step - Data Generation
#########################################################################
#########################################################################
#########################################################################

p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['problem'] = 'tensile_test'      #'cantilever_beam' #
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
p['nu'] = 0.28 #0.3
p['E'] = 210e9 #1e5
#p['K_torsion'] = 1e15
p['k_x'] = 1e15
p['k_y'] = 1e13

""" p['constitutive'] = 'orthotropic'
p['E_m'] = 1e5
p['E_d'] = 0.
p['nu_12'] = 0.3
p['G_12'] = p['E_m']/(2*(1+p['nu_12'])) """


p['length'] = 5
p['breadth'] = 1
p['load'] = [1000,0] #[0.1,0] 
# N/m², m, kg, sec, N
#p['rho'] = 7750
#p['g'] = 9.81
#p['k_x'] = 1e15
#p['k_y'] = 1e13
#p['K_torsion'] = 1e11

#print(p['g']*0.2*p['rho'])
# MPa, mm, kg, sec, N
#p['rho'] = 7750e-9 #kg/mm³
#p['g'] = 9.81#e3 #mm/s² for units to be consistent g must be given in m/s².
#p['E'] = 210e3 #N/mm² or MPa
#p['length'] = 1000
#p['breadth'] = 200
#p['load'] = 100e-6 #N/mm²

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
problem.add_sensor(fenicsX_concrete.sensors.ReactionForceSensor())
problem.solve() 
reaction_force_data = problem.sensors['ReactionForceSensor'].data[-1]
displacement_data = problem.displacement.x.array
problem.pv_plot("Displacement.xdmf")

#########################################################################
#########################################################################
#########################################################################
#1st Step - Setting up inverse problem
#########################################################################
#########################################################################
#########################################################################

p['constitutive'] = 'orthotropic'
p['E_m'] = 210e9 #1e5
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  p['E_m']/(2*(1+p['nu_12'])) #(0.5*1e5)/(1+0.3)

""" p['constitutive'] = 'isotropic'
p['nu'] = 0.3
p['E'] = 1e5 """

problem = fenicsX_concrete.LinearElasticity(experiment, p)
problem.solve_inverse_problem(displacement_data,reaction_force_data)

from numpy import linalg as LA

def forward_model_run(parameters):
    # Function to run the forward model
    problem.E_m.value = parameters[0]
    problem.E_d.value = parameters[1]
    problem.nu_12.value = parameters[2]
    problem.G_12.value = parameters[3]

    #problem.lambda_.value = parameters[0] * parameters[1] / ((1.0 + parameters[1]) * (1.0 - 2.0 * parameters[1])) #parameters[0]
    #problem.mu.value = parameters[0] / (2.0 * (1.0 + parameters[1])) #parameters[1]
    
    #problem.E.value = parameters[0]
    #problem.nu.value = parameters[1]

    problem.solve_inverse_problem(displacement_data, reaction_force_data)
    return problem.momentum_balance, problem.force_balance

def cost_function(param):
    # Function to calculate the cost function
    force_balance_free, force_balance_fixed = forward_model_run(param)
    print(LA.norm(force_balance_free) , LA.norm(force_balance_fixed))
    #return LA.norm(momentum_balance)**2 +  LA.norm(force_balance)**2
    return np.dot(force_balance_free, force_balance_free) + np.dot(force_balance_fixed, force_balance_fixed) #+ 0.1*LA.norm(param,1)#

from scipy.optimize import minimize, least_squares
#res = minimize(cost_function, np.array([1e5, 1e2, 0.33, 1e4/(2*(1+0.2)) ]), method='nelder-mead',
#               options={'xatol': 1e-8, 'disp': True}) #1e5, 0., 0.3, (0.5*1e5)/(1+0.3) [1e5, 1e3, 0.2, 3e4]

res = minimize(cost_function, np.array([100e9, 20e9, 0.15, 0.75*p['G_12']]), method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True, 'maxiter':1000}) #[210e9, 0., 0.28, p['E_m']/(2*(1+p['nu_12']))]

print(res.x) 
print(p['G_12'])

""" def forward_model_run_2(parameters):
    # Function to run the forward model
    problem.E_m.value = parameters[0]
    problem.E_d.value = 0.
    problem.nu_12.value = parameters[1]
    problem.G_12.value = parameters[2]

    problem.solve_inverse_problem(displacement_data, reaction_force_data)
    return problem.momentum_balance, problem.force_balance

def cost_function_2(param):
    # Function to calculate the cost function
    momentum_balance, force_balance = forward_model_run_2(param)
    return np.dot(momentum_balance,momentum_balance) +  np.dot(force_balance, force_balance) #+ 0.1*LA.norm(param,1)#

res2 = minimize(cost_function_2, np.array([res.x[0], res.x[2], res.x[3]]), method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True, 'maxiter':1000}) #[210e9, 0., 0.28, p['E_m']/(2*(1+p['nu_12']))]

print(res2.x)  """