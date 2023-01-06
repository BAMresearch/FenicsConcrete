import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
sys.path.append(os.path.dirname(parentdir)+'/surf2stl-python')
import numpy as np
import fenicsX_concrete
from scipy import optimize
#import surf2stl 

def collect_sensor_solutions(model_solution, total_sensors):
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model

# Synthetic data generation
para = fenicsX_concrete.Parameters()  # using the current default values
para['problem'] = 'cantilever_beam' #'tensile_test' #cantilever_beam

# N/m², m, kg, sec, N
para['rho'] = 7750
para['g'] = 9.81
para['E'] = 210e9
para['length'] = 1
para['breadth'] = 0.2
para['load'] = 100

# MPa, mm, kg, sec, N
#para['rho'] = 7750e-9 #kg/mm³
#para['g'] = 9.81#e3 #mm/s² for units to be consistent g must be given in m/s².
#para['E'] = 210e3 #N/mm² or MPa
#para['length'] = 1000
#para['breadth'] = 200
#para['load'] = 100e-6 #N/mm²

para['nu'] = 0.28
para['num_elements_length'] = 30
para['num_elements_breadth'] = 20
para['dim'] = 2
experiment = fenicsX_concrete.concreteSlabExperiment(para)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, para)      # Specifies the material law and weak forms.


sensor = []
#for i in range(20):
#    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0., 0]])))
#    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.05, 0]])))
#    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.15, 0]])))
#    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.2, 0]])))
#    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))
#
#    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0., 0]])))
#    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.25, 0]])))
#    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.75, 0]])))
#    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 1., 0]])))

for i in range(10):
    for j in range(11):
        sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[para['length']/10*(i+1), 0.2, 0]]))) #1/20
        sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[para['length']/10*(i+1), 0, 0]])))

number_of_sensors = len(sensor)

for i in range(number_of_sensors):
    problem.add_sensor(sensor[i])

#mt=MemoryTracker()
problem.solve()
problem.pv_plot()
displacement_data = collect_sensor_solutions(problem.sensors, number_of_sensors)

#Clean the sensor data for the next simulation run
problem.clean_sensor_data()

max_disp_value_ver= np.amax(np.absolute(displacement_data[:,1]))
max_disp_value_hor= np.amax(np.absolute(displacement_data[:,0]))

#sigma_error_hor = 0.05*max_disp_value_hor
#sigma_error_ver = 0.05*max_disp_value_ver
#sigma_prior = 0.1*max_disp_value

np.random.seed(42) 
distortion_hor = np.random.normal(0, 1e-8, (number_of_sensors)) #0.05
distortion_ver = np.random.normal(0, 1e-8, (number_of_sensors)) #0.05

displacement_measured_hor = displacement_data[:,0] + distortion_hor
displacement_measured_ver = displacement_data[:,1] + distortion_ver

displacement_measured = np.stack((displacement_measured_hor, displacement_measured_ver), axis = -1)

def forward_model_run(param1, param2, ndim):
    problem.lambda_.value = param1 * param2 / ((1.0 + param2) * (1.0 - 2.0 * param2))
    problem.mu.value = param1 / (2.0 * (1.0 + param2))
    #problem.lambda_ = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    #problem.mu = param_vector[0] / (2.0 * (1.0 + param_vector[1]))

    #print(help(problem.weak_form_problem.A))
    #print(problem.weak_form_problem.A.getValues(range(8),range(8)))

    problem.solve() 

    #mt("MCMC run")
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    if ndim == 1:
        return model_data[:,1]
    if ndim == 2:
        return model_data

#Cost function plot
def cost_func_deterministic(optimised_parameters, measured_data, regularisation_constant = 0.):
    ndim = measured_data.ndim

    #Cost function evaluation
    sim_output = forward_model_run(optimised_parameters[0], optimised_parameters[1], ndim)

    delta = sim_output - measured_data

    parameters_vector = np.array([optimised_parameters[0], optimised_parameters[1]])
    
    #normed_vector = parameters_vector/np.linalg.norm(parameters_vector) #parameters_vector #
    #normed_vector = np.array([optimised_parameters[0]/100, (optimised_parameters[1]-0.2)/0.2])
    #normed_vector = np.array([optimised_parameters[0]/100, (optimised_parameters[1])/0.5])

    if ndim == 2:
        norm_vec_delta = np.linalg.norm(delta, axis=1)
        #norm_vec_delta = np.linalg.norm(delta[:,0])
        #x_component = np.abs(delta[:,0])
        #y_component = np.abs(delta[:,1])
        #print(norm_vec_delta - x_component, '\n')
        #print(abc, '\n')
        #print(norm_vec_delta - y_component)
    else:
        norm_vec_delta = np.copy(delta)
    
    #first_term = np.dot(norm_vec_delta, norm_vec_delta)
    #second_term = regularisation_constant*np.dot(normed_vector,normed_vector)
    #cost_function_value = np.dot(norm_vec_delta, norm_vec_delta) + regularisation_constant*np.dot(normed_vector,normed_vector)
    #print(parameters_vector, cost_function_value) 

    delta_disp = delta.flatten()
    cost_function_value = np.dot(delta_disp, delta_disp) + 0 #regularisation_constant*np.dot(normed_vector,normed_vector)

    return cost_function_value


import matplotlib.pyplot as plt
from matplotlib import cm

def cost_function_plot():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #counter=0
    #E_buildup = np.linspace(50,200,100)
    E_values = np.linspace(175e9,225e9,100)
    #E_values = np.linspace(150e3,250e3,100)
    nu_values = np.linspace(0,0.45,15)
    E_buildup, nu_buildup = np.meshgrid(E_values, nu_values)
    cost_func_val = np.zeros((E_buildup.shape[0],E_buildup.shape[1]))
    for i in range(E_buildup.shape[0]):
        for j in range(nu_buildup.shape[1]):
            cost_func_val[i,j] = cost_func_deterministic(np.array([E_buildup[i,j],nu_buildup[i,j]]), displacement_measured)

    # Plot the surface.
    surf = ax.plot_surface(E_buildup, nu_buildup, cost_func_val,  cmap=cm.coolwarm, edgecolor = 'black',
                           linewidth=0, antialiased=False)
 
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    #surf2stl.write('unique1.stl', E_buildup, nu_buildup, cost_func_val)


#Inverse Problem
def cost_func_and_jac(optimised_parameters, measured_data, regularisation_constant = 0.):
    #Cost function evaluation
    ndim = measured_data.ndim
    sim_output = forward_model_run(optimised_parameters[0], optimised_parameters[1], ndim)

    delta = sim_output - measured_data

    if ndim == 2:
        norm_vec_delta = np.linalg.norm(delta, axis=1)
    else:
        norm_vec_delta = np.copy(delta)

    parameters_vector = np.array([optimised_parameters[0], optimised_parameters[1]])

    cost_function_value = np.dot(norm_vec_delta, norm_vec_delta) + regularisation_constant*np.dot(parameters_vector,parameters_vector)

    
    #Jacobian evaluation
    delta_E = 0.001
    measured_Eplus = forward_model_run(optimised_parameters[0] + delta_E, optimised_parameters[1])
    measured_Eminus = forward_model_run(optimised_parameters[0] - delta_E, optimised_parameters[1])
    derv_E = (measured_Eplus[:,1] - measured_Eminus[:,1])/(2*delta_E)

    delta_nu = 0.0001
    measured_nuplus = forward_model_run(optimised_parameters[0], optimised_parameters[1] + delta_nu)
    measured_numinus = forward_model_run(optimised_parameters[0], optimised_parameters[1] - delta_nu)
    derv_nu = (measured_nuplus[:,1]  - measured_numinus[:,1])/(2*delta_nu)
    print('jacobian called')
    jacobian_value1= 2*np.dot(delta[:,1],derv_E)
    jacobian_value2= 2*np.dot(delta[:,1],derv_nu)
    jacobian_value = np.array([jacobian_value1, jacobian_value2])
    return cost_function_value, jacobian_value

def hessian_function(optimised_parameters, measured_data=displacement_measured):
    #Cost function evaluation
    predicted_displacement = forward_model_run(optimised_parameters[0], optimised_parameters[1])

    delta_E = 0.001
    measured_Eplus = forward_model_run(optimised_parameters[0] + delta_E, optimised_parameters[1])
    measured_Eminus = forward_model_run(optimised_parameters[0] - delta_E, optimised_parameters[1])

    derv_E = (measured_Eplus[:,1]  - measured_Eminus[:,1])/(2*delta_E)
    double_derv_E = (measured_Eplus[:,1]  - 2*predicted_displacement[:,1]  + measured_Eminus[:,1] )/delta_E**2

    delta_nu = 0.0001
    measured_nuplus = forward_model_run(optimised_parameters[0], optimised_parameters[1] + delta_nu)
    measured_numinus = forward_model_run(optimised_parameters[0], optimised_parameters[1] - delta_nu)

    derv_nu = (measured_nuplus[:,1]  - measured_numinus[:,1])/(2*delta_nu)
    double_derv_nu = (measured_nuplus[:,1]  - 2*predicted_displacement[:,1]  + measured_numinus[:,1] )/delta_nu**2

    measured_Eplus_nuplus = forward_model_run(optimised_parameters[0] + delta_E, optimised_parameters[1] + delta_nu)

    measured_Eminus_numinus = forward_model_run(optimised_parameters[0] - delta_E, optimised_parameters[1] - delta_nu)

    measured_Eplus_numinus = forward_model_run(optimised_parameters[0] + delta_E, optimised_parameters[1] - delta_nu)

    measured_Eminus_nuplus = forward_model_run(optimised_parameters[0] - delta_E, optimised_parameters[1] + delta_nu)

    mixed_derv_E_nu = (measured_Eminus_numinus[:,1]  + measured_Eplus_nuplus[:,1]  - measured_Eplus_numinus[:,1]  - measured_Eminus_nuplus[:,1] )/(4*delta_E*delta_nu)

    hessian = np.zeros((2,2))
    delta = predicted_displacement - measured_data
    #print('hessian called')
    hessian[0][0] = 2*(np.dot(derv_E,derv_E) + np.dot(delta[:,1],double_derv_E))
    hessian[1][1] = 2*(np.dot(derv_nu,derv_nu) + np.dot(delta[:,1],double_derv_nu))
    hessian[0][1] = 2*(np.dot(derv_E,derv_nu) + np.dot(delta[:,1],mixed_derv_E_nu))
    hessian[1][0] = 2*(np.dot(derv_E,derv_nu) + np.dot(delta[:,1],mixed_derv_E_nu))
    return hessian


#Probabilistic
def cost_func_probablisitic(optimised_parameters, measured_data, std_dev_error, _sigma_prior):
#def cost_func_probablisitic(optimised_parameters, measured_data, std_dev_error=0.05*max_disp_value_ver, _sigma_prior=sigma_prior):
    #Cost function evaluation
    sim_output = forward_model_run(optimised_parameters[0], optimised_parameters[1])

    delta = sim_output - measured_data

    parameters_vector = np.array([optimised_parameters[0], optimised_parameters[1]])
    
    normed_vector = parameters_vector/np.linalg.norm(parameters_vector)

    norm_vec_delta = np.linalg.norm(delta, axis=1)
    first_term = np.dot(norm_vec_delta, norm_vec_delta)
    second_term = std_dev_error**2*np.dot(parameters_vector, np.linalg.solve(_sigma_prior, parameters_vector))
    cost_function_value = np.dot(norm_vec_delta, norm_vec_delta) + std_dev_error**2*np.dot(parameters_vector, np.linalg.solve(_sigma_prior, parameters_vector))

    return cost_function_value 


#Deterministic
result = optimize.minimize(cost_func_deterministic, np.array([80, 0.25]), args=(displacement_measured,),  method='powell') #Newton-CG
#result = optimize.minimize(cost_func_and_jac, (80, 0.15), args=(displacement_measured), jac=True, hess=hessian_function, method='Newton-CG') #Newton-CG
print(result)

#Constrained Optimisation
#from scipy.optimize import Bounds
#paramter_bounds = Bounds([0.,0.], [np.inf,0.45])
#result = optimize.minimize(cost_func_deterministic, np.array([75, 0.15]), options={'verbose': 1}, bounds=paramter_bounds, args=(displacement_measured),  method='trust-constr') #SLSQP
#print(result.x)

cost_function_plot()
#Probabilistic (MAP)
""" import math
sigma_error = 0.1*math.sqrt(max_disp_value_ver**2 + max_disp_value_hor**2)
sigma_prior= np.zeros((2,2))
sigma_prior[0,0] = 2.5**2
sigma_prior[1,1] = 0.075**2
result = optimize.minimize(cost_func_probablisitic, np.array([95, 0.18]), args=(displacement_measured, sigma_error, sigma_prior),  method='powell')
print(result) """

#Constrained Optimisation