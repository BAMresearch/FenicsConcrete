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
mydict = {
    "test1": [{"name"    : "E",
                "tex"     : "E",  
                "info"    : "Young's Modulus of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 1}]},
                 
                {"name"   : "nu",
                "tex"     : "\nu",  
                "info"    : "Poisson's Ratio of the material",
                "domain"  : None,
                "prior"   : ['Uniform', {'low': 0, 'high': 0.45}]}
                ] 
        
    }  
            

json_string = json.dumps(mydict , indent = 2)
with open('mydata.json', 'w') as f:
    f.write(json_string) 

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

