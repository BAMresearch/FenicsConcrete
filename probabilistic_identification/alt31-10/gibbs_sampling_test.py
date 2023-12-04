import numpy as np
import matplotlib.pyplot as plt
import emcee

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot


# ground truth that is to be inferred
a_true = 8
b_true = 10

# settings for data generation
n_tests = 50
seed = 1
mean_noise = 0.0
std_noise = 1 # is assumed to be known and is not inferred.

# generate the data
np.random.seed(seed)
x_test = np.linspace(-4.0, 4.0, n_tests)
y_true = a_true * x_test + b_true #a_true * x_test**2 #+ b_true * x_test + c_true #a_true * x_test**2 + 
y_test = y_true + np.random.normal(loc=mean_noise, scale=std_noise, size=n_tests)

""" plt.plot(x_test, y_test, "o", label="generated data points")
plt.plot(x_test, y_true, label="ground-truth model")
#plt.plot(x_test, - 2*x_test , label="ground-truth model") #1*np.exp(x_test)
plt.title("Data vs. ground truth")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show() """

import json
with open('probabilistic_identification/parameters_gibbs.json', 'r') as f: 
    json_object = json.loads(f.read()) 

nwalkers = 4
ndim = 2 #len(json_object.get('parameters')) 
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

# Correction in initial sampling of spike sla required
from scipy.stats import invgamma, halfcauchy, norm, bernoulli

def log_probability_beta(theta, _x_test, _y_test, _std_noise):
    lp = 0
    lp += bernoulli.logpmf(theta[0], p = 0.5)

    if theta[0] == 0:
        lp += norm.logpdf(theta[1], loc = 0, scale = 0.01)
    elif theta[0] == 1:
        lp += norm.logpdf(theta[1], loc = 0, scale = 4)

    y_model = theta[1] * _x_test + 10

    lp += -0.5 * np.sum((_y_test - y_model) ** 2 / _std_noise**2 + np.log(_std_noise**2))

    if not np.isfinite(lp):
        return -np.inf
    return lp 


def log_probability_lambda(theta, _x_test, _y_test, _std_noise):
    lp = 0
    lp += bernoulli.logpmf(theta[0], p = 0.5)

    if theta[0] == 0:
        lp += norm.logpdf(theta[1], loc = 0, scale = 0.01)
    elif theta[0] == 1:
        lp += norm.logpdf(theta[1], loc = 0, scale = 4)

    y_model = theta[1] * _x_test + 10

    lp += -0.5 * np.sum((_y_test - y_model) ** 2 / _std_noise**2 + np.log(_std_noise**2))

    if not np.isfinite(lp):
        return -np.inf
    return lp 

sampler1 = emcee.EnsembleSampler(nwalkers, ndim, log_probability_beta, args=(x_test, y_test, std_noise), moves=[emcee.moves.GaussianMove(1)])
sampler1.run_mcmc(start_parameters, 1, progress=False)
samples = sampler1.get_chain()
print(1)