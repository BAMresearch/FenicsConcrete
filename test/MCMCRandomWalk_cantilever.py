import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import math
import matplotlib.pyplot as plt

#Original target density plot ---------------------------------------------------------------------------------------------------------------

def collect_sensor_solutions(model_solution, total_sensors):
    """Returns the displacement values given by the forward model."""
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

sensor = []
for i in range(20):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.1, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.05, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))

number_of_sensors = len(sensor)

for i in range(number_of_sensors):
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


prior_variance = np.array([[math.log(10)**2,0],[0,math.log(0.05*100)**2]])
param_mean = np.array([math.log(95),math.log(0.2*100)])
#prior_variance = np.array([[20**2,0],[0,0.1**2]])
#param_mean = np.array([95,0.2])
measurement_err_stddev = 0.2


def log_target_density(param_vector, stddev_measurement_error=measurement_err_stddev, obs_data=observed_data[:,1], _param_mean=param_mean, var_prior=prior_variance):
    
    #Simulation run
    #problem.lambda_.value = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    #problem.mu.value = param_vector[0] / (2.0 * (1.0 + param_vector[1]))
    
    #print(math.exp(param_vector[0]), math.exp(param_vector[1]))
    problem.lambda_.value = math.exp(param_vector[0]) * 0.01 * math.exp(param_vector[1]) / ((1.0 + 0.01 * math.exp(param_vector[1])) * (1.0 - 2.0 * 0.01 * math.exp(param_vector[1])))
    problem.mu.value = math.exp(param_vector[0]) / (2.0 * (1.0 + 0.01 * math.exp(param_vector[1])))

    #problem.lambda_.value = math.exp(param_vector[0])  * math.exp(param_vector[1]) / ((1.0 + math.exp(param_vector[1])) * (1.0 - 2.0 * math.exp(param_vector[1])))
    #problem.mu.value = math.exp(param_vector[0]) / (2.0 * (1.0 +  math.exp(param_vector[1])))

    problem.solve() 
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    displacement_data_vertical = model_data[:,1]
    problem.clean_sensor_data()
    
    #Evaluation of target density
    num_obs = displacement_data_vertical.shape[0]
    num_param =  var_prior.shape[0]
    delta_disp_data= obs_data - displacement_data_vertical
    delta_param = param_vector - _param_mean

    #pdf_likelihood = 2*math.pi**(-0.5*num_obs)*(stddev_measurement_error**-num_obs)*math.exp(-np.dot(delta_disp_data,delta_disp_data)/(2*stddev_measurement_error**2)) 
    #pdf_prior = 2*math.pi**(-0.5*num_param)*np.linalg.det(var_prior)**(-0.5)*np.exp(-np.dot(delta_param, np.linalg.solve(prior_variance, delta_param)))

    pdf_likelihood = -num_obs*math.log(stddev_measurement_error) - 0.5*(np.dot(delta_disp_data,delta_disp_data)/stddev_measurement_error**2) 
    #print(math.exp(pdf_likelihood))
    #if math.isnan(pdf_likelihood):
    #    pdf_likelihood = 0
    pdf_prior = 0#-0.5*math.log(np.prod(np.diagonal(var_prior))) - 0.5*(np.dot(delta_param, np.linalg.solve(var_prior, delta_param)))

    #del displacement_data_vertical
    #print(2)
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

#num_samples = 5000
#sigma = 0.35
#Metropolis_chain, accepted_proposals = metropolis_hastings(np.array([3,5]), num_samples, 2, sigma)
E_start = np.random.uniform(math.log(50),math.log(150),size=1)
nu_start = np.random.uniform(math.log(0.2*100),math.log(0.4*100),size=1)
num_samples_burnin = 7500
sigma =     math.sqrt(0.045)#0.37
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
