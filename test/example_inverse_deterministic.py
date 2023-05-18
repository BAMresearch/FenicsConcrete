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

# N/m², m, kg, sec, N
#p['length'] = 5
#p['breadth'] = 1
#p['load'] = [1e6,0] #[0, -10] #
#p['rho'] = 7750
#p['g'] = 9.81
#p['E'] = 210e9 

#p['k_x'] = 1e6
#p['k_y'] = 1e8
#p['K_torsion'] = 1e11


# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3,0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa


experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
#problem.add_sensor(fenicsX_concrete.sensors.ReactionForceSensor())
problem.solve() 
#reaction_force_data = problem.sensors['ReactionForceSensor'].data[-1]
displacement_data = problem.displacement.x.array
#problem.pv_plot("Displacement_cantilever.xdmf")

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

def forward_model_run(parameters):
    # Function to run the forward model
    #problem.E.value = parameters[0]
    #problem.nu.value = parameters[1]
    problem.E_m.value = parameters[0]
    problem.E_d.value = parameters[1]
    problem.nu_12.value = parameters[2]
    problem.G_12.value = parameters[3]
    problem.k_x.value = 1e12            #parameters[4]               #abs(1/parameters[0] -1) +1e9 #*p['E_m']
    problem.k_y.value = 1e12            #parameters[5]               #abs(1/parameters[1] -1) +1e9#*p['E_m']
    print(parameters[0:4])
    problem.solve()
    return problem.displacement.x.array


from numpy import linalg as LA
def cost_function(param):
    # Function to calculate the cost function
    displacement_model = forward_model_run(param)  
    delta_displacement = displacement_model - displacement_data
    #print(delta_displacement)
    #print('Optimisation Parameters',param[0], param[1])
    print('Cost Function', np.dot(delta_displacement, delta_displacement))
    return np.dot(delta_displacement, delta_displacement)  

from scipy.optimize import minimize, least_squares


start_point = np.array([100e6, 20e6, 0.15, 0.6*p['E_m']/(2*(1+p['nu_12']))]) #, 1e8, 1e8
parameter_bounds = [(0, np.inf), (0, np.inf), (0, 0.45), (0, np.inf)] #, (0, np.inf), (0, np.inf)
res = minimize(cost_function, start_point, method='Nelder-Mead', bounds=parameter_bounds,#0.50.5
              options={'disp': True, 'maxiter':400}) 
print(res.x) 
print(p['G_12'])

# res = least_squares(cost_function, np.array([51e6]), bounds=parameter_bounds, verbose=0, ftol=1e-8, gtol=1e-20)
#start_point = np.array([50e9, 20e9, 0.15, 0.6*p['E_m']/(2*(1+p['nu_12']))]) #, 1e8, 1e8
#parameter_bounds = [(0, np.inf), (0, np.inf), (0, 0.45), (0, np.inf)] #, (0, np.inf), (0, np.inf)

""" #Isotropic data, istropic identification of tensile test
start_point = np.array([50e9, 0.15]) #, 1e8, 1e8
parameter_bounds = [(0, np.inf), (0, 0.45)] #, (0, np.inf), (0, np.inf)
res = minimize(cost_function, start_point, method='Nelder-Mead', bounds=parameter_bounds,#0.50.5
              options={'disp': True, 'maxiter':400}) 
print(res.x) 
print(p['G_12']) """