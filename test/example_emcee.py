
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
#import matplotlib.pyplot as plty


def collect_sensor_solutions(model_solution, total_sensors):
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model

# Synthetic data generation
para = fenicsX_concrete.Parameters()  # using the current default values
para['length'] = 1
para['breadth'] = 0.2
para['num_elements_length'] = 20
para['num_elements_breadth'] = 10
para['dim'] = 2
para['bc_setting'] = 'free'
para['mesh_density'] = 10
experiment = fenicsX_concrete.concreteSlabExperiment(para)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, para)      # Specifies the material law and weak forms.

number_of_sensors = 20
sensor = []
for i in range(number_of_sensors):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.1, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))

for i in range(len(sensor)):
    problem.add_sensor(sensor[i])

#mt=MemoryTracker()
problem.solve()
displacement_data = collect_sensor_solutions(problem.sensors, number_of_sensors)

#Clean the sensor data for the next simulation run
problem.clean_sensor_data()

max_disp_value = np.amax(np.absolute(displacement_data[:,1]))


sigma_error = 0.1*max_disp_value

np.random.seed(42) 
distortion = np.random.normal(0, sigma_error, (number_of_sensors,2))

observed_data = displacement_data + distortion


prior_variance = np.array([[20**2,0],[0,0.1**2]])
param_mean = np.array([85,0.2])
measurement_err_stddev = 0.5

""" def func(param_vector):
    problem.lambda_.value = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    problem.mu.value = param_vector[0] / (2.0 * (1.0 + param_vector[1]))
    #problem.lambda_ = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    #problem.mu = param_vector[0] / (2.0 * (1.0 + param_vector[1]))
    problem.solve() 
    #mt("MCMC run")
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()

#func(np.array([100, 0.2]))
func(np.array([150, 0.23]))
func(np.array([0, 0]))
print('e')
"""
import math
def log_target_density(param_vector, stddev_measurement_error=measurement_err_stddev, obs_data=observed_data[:,1], param_mean=param_mean, var_prior=prior_variance):
    
    #Simulation run
    problem.lambda_.value = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    problem.mu.value = param_vector[0] / (2.0 * (1.0 + param_vector[1]))
    #problem.lambda_ = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    #problem.mu = param_vector[0] / (2.0 * (1.0 + param_vector[1]))
    problem.solve() 
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    
    #Evaluation of target density
    displacement_data_vertical = model_data[:,1]
    num_obs = displacement_data_vertical.shape[0]
    num_param =  var_prior.shape[0]
    delta_disp_data= obs_data - displacement_data_vertical
    delta_param = param_vector - param_mean

    #pdf_likelihood = 2*math.pi**(-0.5*num_obs)*(stddev_measurement_error**-num_obs)*math.exp(-np.dot(delta_disp_data,delta_disp_data)/(2*stddev_measurement_error**2)) 
    #pdf_prior = 2*math.pi**(-0.5*num_param)*np.linalg.det(var_prior)**(-0.5)*np.exp(-np.dot(delta_param, np.linalg.solve(prior_variance, delta_param)))

    pdf_likelihood = 2*math.pi**(-0.5*num_obs)*(stddev_measurement_error**-num_obs)*(-np.dot(delta_disp_data,delta_disp_data)/(2*stddev_measurement_error**2)) 
    pdf_prior = 2*math.pi**(-0.5*num_param)*np.linalg.det(var_prior)**(-0.5)*(-0.5*np.dot(delta_param, np.linalg.solve(prior_variance, delta_param)))
    #del displacement_data_vertical
    #print(2)
    return pdf_likelihood + pdf_prior

import emcee
ndim = 2
#p_vector=np.array([[90.5,0.17], [87,0.4], [85,0.3],[92,0.4], [81,0.3],[20,0.4]])
nwalkers = 40
p_vector = np.random.rand(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_target_density)
#sampler.run_mcmc(p_vector, 100, progress=True)
state=sampler.run_mcmc(p_vector, 2000, progress=True)
sampler.reset()
sampler.run_mcmc(state, 10000, progress=True)


import plotly.express as px
samples = sampler.get_chain(flat=True)
print(samples.shape)
fig = px.histogram(samples[:, 0], nbins=None)
fig.show()

fig1 = px.histogram(samples[:, 1], nbins=None)
fig1.show()
np.savetxt("output", samples, fmt='%.8e')