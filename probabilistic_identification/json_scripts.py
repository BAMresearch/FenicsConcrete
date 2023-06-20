import json
""" mydict = {
    "test1": {"E_m" : 210*10**6,
              "E_d" : 70*10**9,} , 
              
    "test2": {"E_m" : 210*10**6,
                "E_d" : 70*10**9,} ,

    "test3": {"E_m" : 210*10**6,       
                "E_d" : 70*10**9,} ,
}
json_string = json.dumps(mydict , indent = 2)
with open('mydata.json', 'w') as f:
    f.write(json_string) """

""" with open('mydata.json', 'r') as f:
    json_object = json.loads(f.read()) 

print(json_object)"""

""" class Person:
    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def print_info(self):
        print("Name: ", self.name)
        print("Age: ", self.age)
        print("Weight: ", self.weight)

    def get_older(self, years):
        self.age += years

    def save_to_json(self, filename):
        person_dict = {'name' : self.name, 'age' : self.age, 'weight' : self.weight}
        with open(filename, 'w') as f:
            f.write(json.dumps(person_dict, indent = 2)) #python dict to json string
    
    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.loads(f.read())   #json string to python dict
        self.name = data['name']
        self.age = data['age']
        self.weight = data['weight']

p1 = Person("John", 36, 80)
p1.print_info()
p1.get_older(4)
p1.save_to_json('person.json')

p2 = Person(None, None, None)
p2.load_from_json('person.json')
p2.print_info() """


import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
#import matplotlib.pyplot as plt
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, LogNormal, Exponential
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (problem solving)
from probeye.inference.scipy.solver import MaxLikelihoodSolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.dynesty.solver import DynestySolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

#r"$E_m$"

""" mydict = {
    "test1": [{"name"    : "E",
                "tex"     : "$E$",  
                "info"    : "Young's Modulus of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 1}]},
                 
                {"name"   : "nu",
                "tex"     : "$\\nu$",  
                "info"    : "Poisson's Ratio of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 0.45}]}
                ] 
        
    }   """

""" mydict = {
    "parameters": [{"name"    : "E_m",
                "tex"     : "$E_m$",  
                "info"    : "Young's Modulus of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 1}]},

                {"name"    : "E_d",
                "tex"     : "$E_d$",  
                "info"    : "Young's Modulus of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 1}]},
                 
                {"name"   : "nu",
                "tex"     : "$\\nu$",  
                "info"    : "Poisson's Ratio of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 0.45}]},

                {"name"   : "G_12",
                "tex"     : "$G_{12}$",  
                "info"    : "Shear Modulus",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 100*10**6}]},

                {"name"   : "k_x",
                "tex"     : "$k_x$",  
                "info"    : "Spring Stiffness in horizontal direction",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 0.45}]},

                {"name"   : "k_y",
                "tex"     : "$k_y$",  
                "info"    : "Spring Stiffness in vertical direction",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 0.45}]}, 
                ],
    "MCMC": {
            "parameter_scaling" : True,
            "nburn": 125,
            "nsteps": 125,
            "pair_plot_name": "pair_plot_scaled_parameters.png",
            "trace_plot_name": "trace_plot_scaled_parameters.png"
          }
        
    }  
            

json_string = json.dumps(mydict , indent = 3)
with open('test_config.json', 'w') as f:
    f.write(json_string) 
 """
""" with open('mydata.json', 'r') as f:
    json_object = json.loads(f.read()) 
    #json_object_2 = json.load(f)


def prior_func(para :list):
    if para[0] == 'Uniform':
        return Uniform(low = para[1]['low'], high = para[1]['high'])
    elif para[0] == 'Normal':
        return Normal(mu = para[1]['mu'], sigma = para[1]['sigma'])
    elif para[0] == 'LogNormal':
        return LogNormal(mu = para[1]['mu'], sigma = para[1]['sigma'])
    elif para[0] == 'Exponential':
        return Exponential(scale = para[1]['scale'], shift = para[1]['shift'])
    else:
        raise ValueError("Prior distribution not implemented")

ProbeyeProblem = InverseProblem("My Problem")
for test, parameters in json_object.items():
    for parameter in parameters:
        #print(parameter)
        ProbeyeProblem.add_parameter(name = parameter['name'], 
                                     tex = "r" + parameter['tex'], 
                                     info = parameter['info'], 
                                     domain = parameter['domain'] if parameter['domain'] != None else "(-oo, +oo)",
                                     prior = prior_func(parameter['prior']))

print(ProbeyeProblem.parameters) """


""" def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf = acf / (len(x)*np.ones(len(x)) - np.arange(len(x)))
    #acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf """

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

# Build the celerite model:
import celerite
from celerite import terms

kernel = terms.RealTerm(log_a=0.0, log_c=-6.0)
kernel += terms.RealTerm(log_a=0.0, log_c=-2.0)

# The true autocorrelation time can be calculated analytically:
true_tau = sum(2 * np.exp(t.log_a - t.log_c) for t in kernel.terms)
true_tau /= sum(np.exp(t.log_a) for t in kernel.terms)
true_tau

# Simulate a set of chains:
gp = celerite.GP(kernel)
t = np.arange(2000000)
gp.compute(t)
y = gp.sample(size=32)

# Let's plot a little segment with a few samples:
""" plt.plot(y[:3, :300].T)
plt.xlim(0, 300)
plt.xlabel("step number")
plt.ylabel("$f$")
plt.title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(true_tau), fontsize=14) """

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# Make plots of ACF estimate for a few different chain lengths
window = int(2 * true_tau)
tau = np.arange(window + 1)
f0 = kernel.get_value(tau) / kernel.get_value(0.0)

""" # Loop over chain lengths:
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for n, ax in zip([10, 100, 1000], axes):
    nn = int(true_tau * n)
    ax.plot(tau / true_tau, f0, "k", label="true")
    ax.plot(
        tau / true_tau,
        autocorr_func_1d(y[0, :nn])[: window + 1],
        label="estimate",
    )
    ax.set_title(r"$N = {0}\,\tau_\mathrm{{true}}$".format(n), fontsize=14)
    ax.set_xlabel(r"$\tau / \tau_\mathrm{true}$")

axes[0].set_ylabel(r"$\rho_f(\tau)$")
axes[-1].set_xlim(0, window / true_tau)
axes[-1].set_ylim(-0.05, 1.05)
axes[-1].legend(fontsize=14) """

# Compute the estimators for a few different chain lengths

N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 10)).astype(int)

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y: #for each chain
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def tau_calculation(y):
    return 1+ 2*np.sum(autocorr_func_1d(y)[:822])

new = np.empty(len(N))

for i, n in enumerate(N):
    #gw2010[i] = autocorr_gw2010(y[:, :n])
    new[i] = autocorr_new(y[:, :n])

plt.loglog(N, new, "o-", label="new")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.axhline(true_tau, color="k", label="truth", zorder=-100)
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14);
plt.show()