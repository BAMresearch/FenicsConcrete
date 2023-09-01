
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
import fenicsX_concrete
import json #math
#from scipy import optimize
import pandas
import matplotlib.pyplot as plt
import scipy.stats as stats

with open('probabilistic_identification/sim_output/sensor_data.json', 'r') as f: 
    json_object = json.loads(f.read()) 

parameters_list = ["E"] #"sigma" , "nu"
posterior_data = np.loadtxt("probabilistic_identification/sim_output/posterior.csv", delimiter=',')
posterior_data = posterior_data.reshape(posterior_data.shape[0], posterior_data.shape[1]// len(parameters_list), len(parameters_list))
#posterior_data = np.transpose(posterior_data, (1,0,2))


    
def combine_test_results(test_results):
    if len(test_results) == 1:
        return test_results[0]
    else:
        return np.concatenate((test_results[0], combine_test_results(test_results[1:])))

def add_noise_to_data(clean_data, no_of_sensors):
    max_disp = np.amax(np.absolute(clean_data))
    min_disp = np.amin(np.absolute(clean_data))
    print('Max', max_disp, 'Min', min_disp)
    if json_object.get('MCMC').get('Error'):
        return clean_data + np.random.normal(0, 0.01 * min_disp, no_of_sensors) ################################################################
    else:
        return clean_data

#############################################################################################################################
#############################################################################################################################
#1st Step - Forward Model Runs
#############################################################################################################################
#############################################################################################################################

p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['problem'] =  'tensile_test'    #'tensile_test' #'bending_test' #bending+tensile_test
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


p['constitutive'] = 'isotropic' #'orthotropic' 
p['nu'] = 0.28#0.28 

# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6#210e6 #Kgmms⁻2/mm² 

p['dirichlet_bdy'] = 'left'

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

#Sparse data (with sensors)

sensor = []
for i in json_object:
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([json_object.get(i).get('where')]), json_object.get(i).get('alphabetical_position'))) #1/20

for i in range(len(sensor)):
    problem.add_sensor(sensor[i])

problem.experiment.bcs = problem.experiment.create_displ_bcs(problem.experiment.V)
problem.apply_neumann_bc()
problem.calculate_bilinear_form()

E = np.array([210e6, 1.896897616300463974e+08])
for i in range(2):
    problem.E.value = E[i]
    problem.solve()
    problem.pv_plot("Displacement.xdmf")
