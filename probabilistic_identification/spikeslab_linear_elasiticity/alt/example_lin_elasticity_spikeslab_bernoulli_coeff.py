
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
#import matplotlib.pyplot as plt
import fenicsX_concrete
import json #math
import pandas as pd
#from scipy import optimize

with open('probabilistic_identification/test_config.json', 'r') as f: 
    json_object = json.loads(f.read()) 

# Adding sensors to the problem definition.
def add_sensor(_problem, _dirichlet_bdy, _sensors_num_edge_hor, _sensors_num_edge_ver): 
    sensor = []
    if _dirichlet_bdy == 0: #'left'
        for i in range(_sensors_num_edge_hor): 
            #print((p['length']*(i+1))/_sensors_num_edge_hor) #p['length']
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[(p['length']*(i+1))/_sensors_num_edge_hor, 0, 0]]), 'top')) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[(p['length']*(i+1))/_sensors_num_edge_hor, p['breadth'], 0]]), 'bottom'))
        
        for i in range(_sensors_num_edge_ver):
            #print((p['breadth']*(i+1))/(_sensors_num_edge_ver+1))
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length'], (p['breadth']*(i+1))/(_sensors_num_edge_ver+1), 0]]), 'right'))

        for i in range(len(sensor)):
            _problem.add_sensor(sensor[i])
        return len(sensor)
    
def store_sensor_data(_problem):
    mydict = {}
    for i in _problem.sensors:
       sensor = {i :    
        {"alphabetical_position" : problem.sensors[i].alphabetical_position,
         "where" : problem.sensors[i].where[0].tolist(),
         "data" : problem.sensors[i].data[0].tolist()}
        } 
       mydict.update(sensor)
    json_string = json.dumps(mydict , indent = 3)
    with open(json_object.get('Data').get('sensor_data'), 'w') as f:
        f.write(json_string) 
    
    
def run_test(exp, prob, dirichlet_bdy, load, sensor_flag = 0):
    if dirichlet_bdy == 0:
        dirichlet_bdy = 'left'
    prob.p.dirichlet_bdy = dirichlet_bdy
    exp.p.dirichlet_bdy = dirichlet_bdy
    prob.p.load = load
    prob.experiment.bcs = prob.experiment.create_displ_bcs(prob.experiment.V)
    prob.apply_neumann_bc()
    prob.calculate_bilinear_form()
    prob.solve()
    prob.pv_plot("Displacement.xdmf")
    #store_sensor_data(prob)
    if sensor_flag == 0:
        return prob.displacement.x.array
    elif sensor_flag == 1 :
        counter=0
        displacement_at_sensors = np.zeros((len(prob.sensors),2))
        for i in prob.sensors:
            displacement_at_sensors[counter] = prob.sensors[i].data[-1]
            counter += 1
        #prob.sensors = fenicsX_concrete.sensors.Sensors()
        return displacement_at_sensors#.flatten()
    
""" def combine_test_results(test_results):
    if len(test_results) == 1:
        return test_results[0]
    else:
        return np.concatenate((test_results[0], combine_test_results(test_results[1:]))) """

def add_noise_to_data(clean_data, no_of_sensors):
    #max_disp = np.amax(np.absolute(clean_data))
    #min_disp = np.amin(np.absolute(clean_data))
    #print('Max', max_disp, 'Min', min_disp)
    #if json_object.get('MCMC').get('Error'):
    #    return clean_data + np.random.normal(0, 0.01 * min_disp, no_of_sensors) ################################################################
    #else:
    return clean_data + np.random.normal(0, 1e-5, no_of_sensors)

#############################################################################################################################
#############################################################################################################################
#1st Step - Data Generation
#############################################################################################################################
#############################################################################################################################

p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['degree'] = 1
p['num_elements_length'] = 25
p['num_elements_breadth'] = 5
p['dim'] = 2
# Uncertainty type:
# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs.
# 3: Torsion Springs
p['uncertainties'] = [0]
#p['k_x'] = 0.5e7
#p['k_y'] = 0.5e7

p['constitutive'] = 'isotropic' #'orthotropic' 
p['nu'] = 0.28

# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 1#1000
p['breadth'] = 0.05#50

p['load'] = [0, -2e7] #[1e3, 0] 
p['lower_limit'] = 0.9*p['length']
p['upper_limit'] = p['length']

p['rho'] = 7750 #7750e-9 #kg/mm³
p['g'] = 9.81 #9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e9 #200e6 #Kgmms⁻2/mm² 

p['dirichlet_bdy'] = 'left'
p['body_force'] = False

sensors_num_edge_hor = 5
sensors_num_edge_ver = 4

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

#Adding sensors to the problem definition.
test1_sensors_total_num = add_sensor(problem, 0, sensors_num_edge_hor, sensors_num_edge_ver)
sensor_positions = np.zeros((test1_sensors_total_num, 3))
counter = 0
for i in problem.sensors:
    sensor_positions[counter] = problem.sensors[i].where[0]
    counter += 1

#Sparse data (with sensors)

temperature_data = np.arange(15, 35, 4) # in degree celsius
youngs_modulus = np.zeros(len(temperature_data))
data = np.zeros((2*test1_sensors_total_num, len(temperature_data)))
for counter, temp in enumerate(temperature_data):
    youngs_modulus[counter] = (235 - 0.04 * temp ** 2)*10**9
    problem.E.value = youngs_modulus[counter] #Remember problem.p.E is still at its initial value.

    #Adding sensors to the problem definition.
    #test1_sensors_total_num = add_sensor(problem, 0, sensors_num_edge_hor, sensors_num_edge_ver)
    #sensor_positions = np.zeros((test1_sensors_total_num, 3))
    #counter = 0
    #for i in problem.sensors:
    #    sensor_positions[counter] = problem.sensors[i].where[0]
    #    counter += 1

    test1_data = run_test(experiment, problem, 0, p['load'] , 1)
    test1_x_component = add_noise_to_data(test1_data[:,0], test1_sensors_total_num)
    test1_y_component = add_noise_to_data(test1_data[:,1], test1_sensors_total_num)

    # Data stored in the form of XYXY components.
    data[:,counter] = np.vstack((test1_x_component, test1_y_component)).T.flatten()

displacement_data = data.flatten('F')

#########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# Inverse Problem Setup
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Loading Prior Distributions from .json File. 

import json
with open('probabilistic_identification/parameters_linear_elasticity_ber.json', 'r') as f: 
    json_object = json.loads(f.read()) 

nwalkers = 6
ndim = len(json_object.get('parameters')) 

from scipy.stats import invgamma, halfcauchy, norm, bernoulli, uniform
start_parameters = np.zeros((nwalkers, ndim))
counter = 0

# This loop reads the parameters from the json file and samples from the prior distributions
for index, parameter in enumerate(json_object.get('parameters')):
    if parameter['prior'][0] == 'Bernoulli':
        start_parameters[:, index] = bernoulli.rvs(p = parameter['prior'][1]["p"], size=nwalkers)
    elif parameter['prior'][0] == 'Spike-Slab':
        for hyperparameter in parameter['hyperparameters']:
            for ind, param in enumerate(json_object.get('parameters')):
                if hyperparameter == param['name']:
                    lmbda = start_parameters[:, ind]
        start_parameters[:, index] = lmbda*norm.rvs(loc = parameter['prior'][1]["mean"], scale = parameter['prior'][1]["variance"], size=nwalkers) 
    elif parameter['prior'][0] == 'Normal': 
        start_parameters[:, index] = norm.rvs(loc = parameter['prior'][1]["mean"], scale = parameter['prior'][1]["variance"], size=nwalkers) 
    elif parameter['prior'][0] == 'Uniform':   
        start_parameters[:, index] = uniform.rvs(loc = parameter['prior'][1]["lower_bound"], scale = parameter['prior'][1]["lower_bound"] + parameter['prior'][1]["upper_bound"], size=nwalkers)   


# Defining the likelihood function

def log_likelihood(theta, displacement_data, _sigma):
    
    #youngs_modulus_proposal = theta[0]*np.ones(len(temperature_data)) + theta[1]*temperature_data + theta[2]*np.square(temperature_data)# + theta[3]*np.log(temperature_data) + theta[4]*np.exp(-temperature_data)
    #if youngs_modulus_proposal.any() <= 5:
    #    return -np.inf

    if theta[0] == 0 and theta[1] == 0:
        return -np.inf
    elif theta[0] == 0 and theta[1] == 1:
        return -np.inf
    
    youngs_modulus_proposal = theta[0]*235*np.ones(len(temperature_data)) -0.04*theta[1]*np.square(temperature_data)# + theta[3]*np.log(temperature_data) + theta[4]*np.exp(-temperature_data)
    displacement_model = np.zeros((2*test1_sensors_total_num, len(temperature_data)))
    for counter, value in enumerate(youngs_modulus_proposal):
        problem.E.value = value*10**9 #Remember problem.p.E is still at its initial value.
        test1_data = run_test(experiment, problem, 0, p['load'] , 1)
        test1_x_component = add_noise_to_data(test1_data[:,0], test1_sensors_total_num)
        test1_y_component = add_noise_to_data(test1_data[:,1], test1_sensors_total_num)
        # Data stored in the form of XYXY components.
        displacement_model[:,counter] = np.vstack((test1_x_component, test1_y_component)).T.flatten()

    displacement_model = displacement_model.flatten('F')

    return -0.5 * np.sum((displacement_data - displacement_model) ** 2 / _sigma**2 + np.log(_sigma**2))

from scipy.stats import invgamma, halfcauchy, norm, bernoulli

def log_prior(theta):
    _lp = 0
    # This loop reads the parameters from the json file and calulates the log prior.
    for index, parameter in enumerate(json_object.get('parameters')):
        if parameter['prior'][0] == 'Bernoulli':
            ## Uncomment the following lines to see the change in trace of the Bernoulli parameters.
            #if theta[index] >= 0.3:
            #    theta[index] = 1
            #else:
            #    theta[index] = 0
            if theta[index] < 0.0 or theta[index] > 1.0:
                return -np.inf
            _lp += bernoulli.logpmf(0 if theta[index] < 0.5 else 1, p = parameter['prior'][1]["p"])            
            #_lp += bernoulli.logpmf(theta[index], p = parameter['prior'][1]["p"])
        #elif parameter['prior'][0] == 'Spike-Slab':
        #    for hyperparameter in parameter['hyperparameters']:
        #        for ind, param in enumerate(json_object.get('parameters')):
        #            if hyperparameter == param['name']:
        #                lmbda = theta[ind]
        #                #if lmbda < 0.0 or lmbda > 1.0:
        #                #    return -np.inf
        #    #_lp += lmbda*norm.logpdf(theta[index], loc = parameter['prior'][1]["mean"], scale = parameter['prior'][1]["variance"]) 
        #    #_lp += lmbda*norm.logpdf(theta[index], loc = parameter['prior'][1]["mean"], scale = parameter['prior'][1]["variance"])
        #    if lmbda >= 0.5:
        #        _lp += norm.logpdf(theta[index], loc = parameter['prior'][1]["mean"], scale = parameter['prior'][1]["variance"])
        #    elif lmbda < 0.5:
        #        theta[index] = 0
        #        _lp += 0
        #    #else:
        #    #    return -np.inf
        
        elif parameter['prior'][0] == 'Normal': 
            _lp[:, index] = norm.logpdf(theta[index], loc = parameter['prior'][1]["mean"], scale = parameter['prior'][1]["variance"]) 

        elif parameter['prior'][0] == 'Uniform':   
            _lp[:, index] = uniform.logpdf(theta[index], loc = parameter['prior'][1]["lower_bound"], scale = parameter['prior'][1]["lower_bound"] + parameter['prior'][1]["upper_bound"])    
    
    return _lp
            

def log_probability(theta, displacement_data, _std_noise):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, displacement_data, _std_noise)

std_noise = 1e-5
import emcee
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(displacement_data, std_noise))
sampler.run_mcmc(start_parameters, 200, progress=True)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["lmbda_a", "lmbda_b"] # Change the labels over here if changes in parameters are made in json file.
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()