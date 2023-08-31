
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



with open('sensor_data.json', 'r') as f: 
    json_object = json.loads(f.read()) 

parameters_list = ["E", "nu", "sigma"] #"E_d",
posterior_data = np.loadtxt("probabilistic_identification/sim_output/posterior.csv", delimiter=',')
posterior_data = posterior_data.reshape(posterior_data.shape[0], posterior_data.shape[1]// len(parameters_list), len(parameters_list))
#posterior_data = np.transpose(posterior_data, (1,0,2))


# Adding sensors to the problem definition.

""" def run_test(exp, prob, dirichlet_bdy, load, sensor_flag = 0):
    #if dirichlet_bdy == 0:
    #    dirichlet_bdy = 'left'
    #prob.p.dirichlet_bdy = dirichlet_bdy
    #exp.p.dirichlet_bdy = dirichlet_bdy
    #prob.p.load = load
    #prob.experiment.bcs = prob.experiment.create_displ_bcs(prob.experiment.V)
    #prob.apply_neumann_bc()
    #prob.calculate_bilinear_form()
    prob.solve()
    prob.pv_plot("Displacement.xdmf")
    if sensor_flag == 0:
        return prob.displacement.x.array
    elif sensor_flag == 1 :
        counter=0
        displacement_at_sensors = np.zeros((len(prob.sensors),2))
        for i in prob.sensors:
            displacement_at_sensors[counter] = prob.sensors[i].data[-1]
            counter += 1
        #prob.clean_sensor_data()
        #prob.sensors = fenicsX_concrete.sensors.Sensors()
        return displacement_at_sensors#.flatten() """
    
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
p['k_x'] = 1e8
p['k_y'] = 1e8

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

""" E = np.array([210e6, 100e6])
nu = np.array([0.28, 0.3])
for i in range(2):
    problem.E.value = E[i]
    problem.nu.value = nu[i]
    problem.solve() """
    #test_data = run_test(experiment, problem, 0, [1e3, 0], 1)
    #print('Test1')

chain_total = posterior_data.shape[1]                   #number of chains
chain_length = posterior_data.shape[0]                  #total chain length
n_tests = 1
#likelihood_predictive = np.zeros((chain_total,chain_length))


chain_total = 2 #posterior_data.shape[1]     
chain_length = posterior_data.shape[0]


for chain_index in range(chain_total): #chain index
    print('Chain #',chain_index,'is being calculated')
    for chain_draw in range(chain_length): #step number of chain/inferred parameters
        problem.E.value = posterior_data[chain_draw,chain_index,0]
        problem.nu.value = posterior_data[chain_draw,chain_index,1]
        sigma = posterior_data[chain_draw,chain_index,2]
        problem.solve()
        for i in problem.sensors:
            problem.sensors[i].data[(chain_index+1)*chain_draw] += problem.sensors[i].data[(chain_index+1)*chain_draw] + np.random.normal(loc=0, scale=sigma)*np.ones(2)


import arviz as az
az.style.use("arviz-doc")
az.rcParams["stats.hdi_prob"] = 0.90
num_plotted_sensors = 0
for i in problem.sensors:
    if 'top' == problem.sensors[i].alphabetical_position:
        num_plotted_sensors += 1
model_data_plot = np.zeros((chain_total,chain_length,num_plotted_sensors))
observed_data_plot = []
sensor_positions = []
axis = 0 # 0 for horizontal, 1 for vertical

sensor_counter = 0
for sensor_key in problem.sensors:
    if 'top' == problem.sensors[sensor_key].alphabetical_position:
        chain_index = 0
        chain_draw = 0
        for data_index in problem.sensors[sensor_key].data:
            model_data_plot[chain_index, chain_draw, sensor_counter] = data_index[axis]
            chain_draw += 1
            if chain_draw == chain_length:
                chain_draw = 0
                chain_index += 1
        sensor_counter += 1

for i in json_object:
    if 'top' == json_object.get(i).get('alphabetical_position'):
            observed_data_plot.append(json_object.get(i).get('data')[axis])
            sensor_positions.append(json_object.get(i).get('where')[axis])

#model_data_plot = np.array(model_data_plot).reshape(-1,chain_length*chain_total)
ax = az.plot_hdi(np.array(sensor_positions), model_data_plot, plot_kwargs={"ls": "--"})
ax.scatter(np.array(sensor_positions), observed_data_plot, color="#b5a7b6", label="generated data points")
plt.show()