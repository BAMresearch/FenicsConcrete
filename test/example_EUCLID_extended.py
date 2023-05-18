import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
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
p['problem'] = 'bending_test'      #'bending_test' #tensile_test
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
#p['k_x'] = 1e15
#p['k_y'] = 1e13
p['rho'] = 7750
p['g'] = 9.81

""" p['constitutive'] = 'orthotropic'
p['E_m'] = 1e5
p['E_d'] = 0.
p['nu_12'] = 0.3
p['G_12'] = p['E_m']/(2*(1+p['nu_12'])) """


p['length'] = 5
p['breadth'] = 1
p['load'] = [1000, 0] #[0.1,0] 
# N/m², m, kg, sec, N
#p['rho'] = 7750
#p['g'] = 9.81
#p['k_x'] = 1e15
#p['k_y'] = 1e13
#p['K_torsion'] = 1e11

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
problem.add_sensor(fenicsX_concrete.sensors.ReactionForceSensor())
problem.solve() 
reaction_force_data = problem.sensors['ReactionForceSensor'].data[-1]
displacement_data = problem.displacement.x.array

dirichlet_dofs = np.concatenate((problem.experiment.bc_x_dof, problem.experiment.bc_y_dof))

distortion = np.random.normal(1e-7, 1e-6 , dirichlet_dofs.shape[0])
displacement_data[dirichlet_dofs] += distortion
problem.pv_plot("Displacement_new.xdmf")

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
p['k_x'] = 1e12#1e15
p['k_y'] = 1e12#1e13
p['uncertainties'] = [0, 2]

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)
problem.solve_inverse_problem(displacement_data,reaction_force_data)

from numpy import linalg as LA

def forward_model_run(parameters):
    # Function to run the forward model
    #problem.E_m.value = parameters[0]
    #problem.E_d.value = parameters[1]
    #problem.nu_12.value = parameters[2]
    #problem.G_12.value = parameters[3]
    #problem.k_x.value = parameters[0]
    #problem.k_y.value = parameters[1]

    #problem.lambda_.value = parameters[0] * parameters[1] / ((1.0 + parameters[1]) * (1.0 - 2.0 * parameters[1])) #parameters[0]
    #problem.mu.value = parameters[0] / (2.0 * (1.0 + parameters[1])) #parameters[1]
    
    #problem.E.value = parameters[0]
    #problem.nu.value = parameters[1]

    problem.solve_inverse_problem(displacement_data, reaction_force_data)
    return problem.momentum_balance, problem.force_balance


def cost_function(param):
    # Function to calculate the cost function
    force_balance_free, force_balance_fixed = forward_model_run(param)  
    print('Params',param[0], param[1])
    return np.dot(force_balance_free, force_balance_free) + np.dot(force_balance_fixed, force_balance_fixed) #+ 0.1*LA.norm(param,1)#

from scipy.optimize import minimize, least_squares
#res = minimize(cost_function, np.array([1e5, 1e2, 0.33, 1e4/(2*(1+0.2)) ]), method='nelder-mead',
#               options={'xatol': 1e-8, 'disp': True}) 


mombal1, bc = forward_model_run(np.array([0., 0.]))
#mombal2, de = forward_model_run(np.array([0., 0.]))
#mombal3, fg = forward_model_run(np.array([1e6, 1e6]))
print(2)

""" res = minimize(cost_function, np.array([1e12, 1e12]), method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True, 'maxiter':1000}) #[210e9, 0., 0.28, p['E_m']/(2*(1+p['nu_12']))] #100e9, 20e9, 0.15, 0.75*p['G_12'],

print(res.x) 
print(p['G_12']) """

