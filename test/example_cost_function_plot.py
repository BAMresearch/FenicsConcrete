import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize

def collect_sensor_solutions(model_solution, total_sensors):
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model

# Synthetic data generation
para = fenicsX_concrete.Parameters()  # using the current default values
para['problem'] = 'tensile_test' #'cantilever_beam'
para['g'] = 9.81e3
para['rho'] = 1
para['E'] = 10e3#100
para['nu'] = 0.2
para['length'] = 1000#1
para['breadth'] = 200#0.2
para['num_elements_length'] = 20
para['num_elements_breadth'] = 10
para['dim'] = 2
para['bc_setting'] = 'free'
para['mesh_density'] = 10
experiment = fenicsX_concrete.concreteSlabExperiment(para)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, para)      # Specifies the material law and weak forms.

sensor = []

for i in range(20):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.2, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.15, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.05, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1000/20*(i+1), 0, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1000/20*(i+1), 50, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1000/20*(i+1), 150, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1000/20*(i+1), 200, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))

number_of_sensors = len(sensor)

for i in range(number_of_sensors):
    problem.add_sensor(sensor[i])

#mt=MemoryTracker()
problem.solve()
displacement_data = collect_sensor_solutions(problem.sensors, number_of_sensors)

#Clean the sensor data for the next simulation run
problem.clean_sensor_data()

#min_disp_value_ver= np.amin(np.absolute(displacement_data[:,1]))
#min_disp_value_hor= np.amin(np.absolute(displacement_data[:,0]))
#sigma_error_hor = 0.1*min_disp_value_hor
#sigma_error_ver = 0.1*min_disp_value_ver
#np.random.seed(42) 


#distortion_hor = np.random.normal(0, 0.0005, (number_of_sensors)) #0.05
#distortion_ver = np.random.normal(0, 0.0005, (number_of_sensors)) #0.05
displacement_measured_hor = displacement_data[:,0] #+ distortion_hor
displacement_measured_ver = displacement_data[:,1] #+ distortion_ver

displacement_measured = np.stack((displacement_measured_hor, displacement_measured_ver), axis = -1)

def forward_model_run(param1, param2, ndim):
    problem.lambda_.value = param1 * param2 / ((1.0 + param2) * (1.0 - 2.0 * param2))
    problem.mu.value = param1 / (2.0 * (1.0 + param2))
    problem.solve() 

    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    if ndim == 1:
        return model_data[:,1]
    if ndim == 2:
        return model_data

#Cost function plot
def cost_func(optimised_parameters, measured_data, regularisation_constant = 0.):
    ndim = measured_data.ndim

    #Cost function evaluation
    sim_output = forward_model_run(optimised_parameters[0], optimised_parameters[1], ndim)

    delta = sim_output - measured_data

    parameters_vector = np.array([optimised_parameters[0], optimised_parameters[1]])
    
    normed_vector = parameters_vector/np.linalg.norm(parameters_vector) #parameters_vector #
    #normed_vector = np.array([optimised_parameters[0]/100, (optimised_parameters[1]-0.2)/0.2])
    normed_vector = np.array([optimised_parameters[0]/100, (optimised_parameters[1])/0.5])

    if ndim == 2:
        norm_vec_delta = np.linalg.norm(delta, axis=1)
    else:
        norm_vec_delta = np.copy(delta)
    
    #cost_function_value = np.dot(norm_vec_delta, norm_vec_delta) + regularisation_constant*np.dot(normed_vector,normed_vector)


    delta_disp = delta.flatten()
    cost_function_value = np.dot(delta_disp, delta_disp) + regularisation_constant*np.dot(normed_vector,normed_vector)

    return cost_function_value

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def cost_function_plot():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #E_buildup = np.linspace(50,200,100)
    #nu_buildup = np.linspace(0,0.45,15)

    E_buildup = np.linspace(2.5e3,20e3,100)
    nu_buildup = np.linspace(0,0.45,15)
    E_buildup, nu_buildup= np.meshgrid(E_buildup, nu_buildup)
    cost_func_val = np.zeros((E_buildup.shape[0],E_buildup.shape[1]))
    for i in range(E_buildup.shape[0]):
        for j in range(nu_buildup.shape[1]):

            cost_func_val[i,j] = cost_func(np.array([E_buildup[i,j],nu_buildup[i,j]]), displacement_measured)

    # Plot the surface.
    surf = ax.plot_surface(E_buildup, nu_buildup, cost_func_val, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

cost_function_plot()
