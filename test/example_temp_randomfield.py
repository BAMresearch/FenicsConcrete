
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
#import matplotlib.pyplot as plt
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, LogNormal
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
from scipy import optimize

def collect_sensor_solutions(model_solution, total_sensors):
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model

def forward_model_run(ndim=2):
    #problem.lambda_.value = param1 * param2 / ((1.0 + param2) * (1.0 - 2.0 * param2))
    #problem.mu.value = param1 / (2.0 * (1.0 + param2))
    #para['E'] = 100e9
    #para['nu'] = 0.21
    lame1, lame2 = problem.get_lames_constants()
    problem.lambda_.vector[:] = lame1
    problem.mu.vector[:] = lame2  #make this vector as a fenics constant array. Update the lame1 and lame2 in each iteration.

    problem.solve() 
    problem.pv_plot()
    #mt("MCMC run")
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    if ndim == 1:
        return model_data[:,1]
    if ndim == 2:
        return model_data

# Synthetic data generation
para = fenicsX_concrete.Parameters()  # using the current default values
para['problem'] = 'cantilever_beam' #'tensile_test' #

# N/m², m, kg, sec, N
para['rho'] = 7750
para['g'] = 9.81
para['E'] = 210e9
para['length'] = 1
para['breadth'] = 0.2
para['load'] = 100
para['K_torsion'] = 1e11

# MPa, mm, kg, sec, N
#para['rho'] = 7750e-9 #kg/mm³
#para['g'] = 9.81#e3 #mm/s² for units to be consistent g must be given in m/s².
#para['E'] = 210e3 #N/mm² or MPa
#para['length'] = 1000
#para['breadth'] = 200
#para['load'] = 100e-6 #N/mm²

para['nu'] = 0.28
para['num_elements_length'] = 30
para['num_elements_breadth'] = 20
para['dim'] = 2
para['uncertainties'] = [1]

experiment = fenicsX_concrete.concreteSlabExperiment(para)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, para)      # Specifies the material law and weak forms.

sensor = []
for i in range(10): #20
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[para['length']/10*(i+1), 0.2, 0]]))) #1/20
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[para['length']/10*(i+1), 0, 0]])))

for i in range(9): #20
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[0, para['breadth']/9*i, 0]]))) #1/20

number_of_sensors = len(sensor)

for i in range(len(sensor)):
    problem.add_sensor(sensor[i])

#mt=MemoryTracker()
problem.solve()
problem.pv_plot()
displacement_data = collect_sensor_solutions(problem.sensors, number_of_sensors)

#Clean the sensor data for the next simulation run
problem.clean_sensor_data()

model_data = forward_model_run()


