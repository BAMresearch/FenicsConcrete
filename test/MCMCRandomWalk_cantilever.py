import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import math
#import matplotlib.pyplot as plt

#Original target density plot ---------------------------------------------------------------------------------------------------------------

def collect_sensor_solutions(model_solution, total_sensors):
    """Returns the displacement values given by the forward model."""
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model

#Forward Model--------------------------------------------------------------------------------------------------------------------------------

def forward_model_run(param1, param2, ndim):
    
    problem.lambda_.value = math.exp(param1) * 0.01 * math.exp(param2) / ((1.0 + 0.01 * math.exp(param2)) * (1.0 - 2.0 * 0.01 * math.exp(param2)))
    problem.mu.value = math.exp(param1) / (2.0 * (1.0 + 0.01 * math.exp(param2)))
    problem.solve() 

    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    if ndim == 1:
        return model_data[:,1]
    if ndim == 2:
        return model_data


# Synthetic data generation -------------------------------------------------------------------------------------------------------------------
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

sensor = []
for i in range(20):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.15, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.05, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))

number_of_sensors = len(sensor)

for i in range(number_of_sensors):
    problem.add_sensor(sensor[i])

#mt=MemoryTracker()
problem.solve()
displacement_data = collect_sensor_solutions(problem.sensors, number_of_sensors)

#Clean the sensor data for the next simulation run
problem.clean_sensor_data()

max_disp_value_ver= np.amax(np.absolute(displacement_data[:,1]))
max_disp_value_hor= np.amax(np.absolute(displacement_data[:,0]))

sigma_error_hor = 0.1*max_disp_value_hor
sigma_error_ver = 0.1*max_disp_value_ver
#sigma_prior = 0.1*max_disp_value

np.random.seed(42) 
distortion_hor = np.random.normal(0, sigma_error_hor, (number_of_sensors))
distortion_ver = np.random.normal(0, sigma_error_ver, (number_of_sensors))

displacement_measured_hor = displacement_data[:,0] + distortion_hor
displacement_measured_ver = displacement_data[:,1] + distortion_ver

displacement_measured = np.stack((displacement_measured_hor, displacement_measured_ver), axis = -1).flatten()
#observed_data = displacement_measured.flatten()

# Probabilistic Inverse Problem Setup
prior_variance = np.array([[math.log(10)**2,0],[0,math.log(0.15*100)**2]])
param_mean = np.array([math.log(95),math.log(0.15*100)])
#prior_variance = np.array([[20**2,0],[0,0.1**2]])
#param_mean = np.array([95,0.2])
int_vec1 = np.stack((np.repeat(0.017**2,number_of_sensors), np.repeat(0.27**2,number_of_sensors)), axis = -1).flatten() #0.023, 0.36
measurement_err_covar = np.diag(int_vec1)
measurement_err_detcov = np.prod(int_vec1)

def log_target_density(param_vector, obs_data=displacement_measured, _measurement_err_covar= measurement_err_covar, _measurement_err_detcov=measurement_err_detcov,  _param_mean=param_mean, var_prior=prior_variance):
    
    #Simulation run
    #problem.lambda_.value = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    #problem.mu.value = param_vector[0] / (2.0 * (1.0 + param_vector[1]))

    #problem.lambda_.value = math.exp(param_vector[0])  * math.exp(param_vector[1]) / ((1.0 + math.exp(param_vector[1])) * (1.0 - 2.0 * math.exp(param_vector[1])))
    #problem.mu.value = math.exp(param_vector[0]) / (2.0 * (1.0 +  math.exp(param_vector[1])))
    
    ndim =  2
    sim_output = forward_model_run(param_vector[0], param_vector[1], ndim)
    delta_disp_data = obs_data - sim_output.flatten()

    #Evaluation of target density
    num_obs = delta_disp_data.shape[0]
    num_param =  var_prior.shape[0]
    delta_param = param_vector - _param_mean

    #pdf_likelihood = 2*math.pi**(-0.5*num_obs)*(stddev_measurement_error**-num_obs)*math.exp(-np.dot(delta_disp_data,delta_disp_data)/(2*stddev_measurement_error**2)) 
    #pdf_prior = 2*math.pi**(-0.5*num_param)*np.linalg.det(var_prior)**(-0.5)*np.exp(-np.dot(delta_param, np.linalg.solve(prior_variance, delta_param)))

    #pdf_likelihood = -num_obs*math.log(measurement_err_detcov) - 0.5*(np.dot(delta_disp_data,delta_disp_data)/stddev_measurement_error**2) 
    pdf_likelihood = -0.5*math.log(_measurement_err_detcov) - 0.5*(np.dot(delta_disp_data,np.linalg.solve(_measurement_err_covar,delta_disp_data)))
    #print(math.exp(pdf_likelihood))
    #if math.isnan(pdf_likelihood):
    #    pdf_likelihood = 0
    pdf_prior = -0.5*math.log(np.prod(np.diagonal(var_prior))) - 0.5*(np.dot(delta_param, np.linalg.solve(var_prior, delta_param)))

    return pdf_likelihood + pdf_prior

#Metropolis Hastings Implementation ----------------------------------------------------------------------------------------------------

def metropolis_hastings(startVector, nsamples, ndim, sigma):
    MH_chain = np.zeros((nsamples, ndim))
    MH_chain[0,:] = startVector 
    counter = 0
    acception_count = 0
    target_density_at_x_n = log_target_density(startVector)

    for counter in range(nsamples-1):
        proposal = MH_chain[counter,:] + np.random.normal(0, sigma**2,2)
 
        log_alpha = log_target_density(proposal) - target_density_at_x_n

        if math.log(np.random.uniform(0,1)) < log_alpha:
            MH_chain[counter+1,:] =  proposal
            target_density_at_x_n = log_target_density(proposal)
            acception_count += 1
        else:
            MH_chain[counter+1,:] = MH_chain[counter,:]

    return MH_chain, acception_count

E_start = np.random.uniform(math.log(85),math.log(105),size=1)
nu_start = np.random.uniform(math.log(0.2*100),math.log(0.4*100),size=1)
num_samples_burnin = 7500
sigma =     0.67 #math.sqrt(0.)#0.37 #0.045
Metropolis_chain, accepted_proposals = metropolis_hastings(np.array([E_start[0],nu_start[0]]), num_samples_burnin, 2, sigma)
print(accepted_proposals)
num_samples = 10000
Metropolis_chain, accepted_proposals = metropolis_hastings(np.array([Metropolis_chain[-1,0],Metropolis_chain[-1,1]]), num_samples, 2, sigma)
print(accepted_proposals)

#Plot Markov Chain ---------------------------------------------------------------------------------------------------------------------------
""" fig.add_trace(go.Scatter3d(
    x=Metropolis_chain[:,0], y=Metropolis_chain[:,1], z=np.zeros((num_samples,)),
    marker=dict(
        size=4,
        #color=z,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=2
    )
))

fig.show() """
import plotly.express as px

for i in range(Metropolis_chain.shape[0]):
    Metropolis_chain[i,0] = math.exp(Metropolis_chain[i,0])
    Metropolis_chain[i,1] = math.exp(Metropolis_chain[i,1])*0.01

fig = px.histogram(Metropolis_chain[:, 0], nbins=None)
fig.show()

fig1 = px.histogram(Metropolis_chain[:, 1], nbins=None)
fig1.show()
# Convergence Measures and their plots ------------------------------------------------------------------------------------------------------

def ergodic_mean(chain):
    mean1 = np.divide(np.cumsum(chain, axis=0)[:,0],np.arange(1,chain.shape[0]+1))
    mean2 = np.divide(np.cumsum(chain, axis=0)[:,1],np.arange(1,chain.shape[0]+1))
    return mean1, mean2

def autocovariance(chain, nsamples, ndim):
    gap = nsamples-1
    autocov = np.zeros((gap, ndim))
    mean = np.sum(chain,axis=0)/nsamples
    variance = np.sum(np.square(chain - mean),axis=0)/nsamples
    #print(variance)
    for j in range(1,gap+1):
        autocov[j-1,:] = np.divide(np.sum(np.multiply((chain[:nsamples-j] - mean),(chain[j::1] - mean)), axis=0),((nsamples-j)*variance))
        #autocov[j-1,1] = np.divide(np.sum(np.multiply((chain[:nsamples-gap] -mean),(chain[gap::1] - mean)), axis=0),((nsamples-j)*variance))
    return autocov

import matplotlib.pyplot as plt
ergo_mean_x1 , ergo_mean_x2 = ergodic_mean(Metropolis_chain)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(np.arange(1,num_samples+1, 1, dtype=None),ergo_mean_x1)
ax1.set_xlabel('Samples') # X-Label
ax1.set_ylabel('Ergodic Mean x1') # Y-Label


ax2.plot(np.arange(1,num_samples+1, 1, dtype=None),ergo_mean_x2)
ax2.set_xlabel('Samples') # X-Label
ax2.set_ylabel('Ergodic Mean x2') # Y-Label
plt.show()


auto_covariance = autocovariance(Metropolis_chain, num_samples, 2)

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(np.arange(1,num_samples, 1, dtype=None),auto_covariance[:,0])
ax1.set_xlabel('gap') # X-Label
ax1.set_ylabel('Autocovariance x1') # Y-Label


ax2.plot(np.arange(1,num_samples, 1, dtype=None),auto_covariance[:,1])
ax2.set_xlabel('gap') # X-Label
ax2.set_ylabel('Autocovariance x2') # Y-Label
plt.show()
