
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
for i in range(20): #20
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.2, 0]]))) #1/20
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[0.25*(i+1), 0.175, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.15, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[0.25*(i+1), 0.10, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[0.25*(i+1), 0.075, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.05, 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))

number_of_sensors = len(sensor)

for i in range(len(sensor)):
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

np.random.seed(42) 
distortion_hor = np.random.normal(0, 0.05, (number_of_sensors))
distortion_ver = np.random.normal(0, 0.05, (number_of_sensors))

displacement_measured_hor = displacement_data[:,0] + distortion_hor
displacement_measured_ver = displacement_data[:,1] + distortion_ver

displacement_measured = np.stack((displacement_measured_hor, displacement_measured_ver), axis = -1)#.flatten()

#max_disp_value = np.amax(np.absolute(displacement_data[:,1]))
#sigma_error = 0.1*max_disp_value

#np.random.seed(42) 
#distortion = np.random.normal(0, sigma_error, (number_of_sensors,2))

#observed_data = displacement_data + distortion

def forward_model_run(param1, param2, ndim=2):
    problem.lambda_.value = param1 * param2 / ((1.0 + param2) * (1.0 - 2.0 * param2))
    problem.mu.value = param1 / (2.0 * (1.0 + param2))
    #problem.lambda_ = param_vector[0] * param_vector[1] / ((1.0 + param_vector[1]) * (1.0 - 2.0 * param_vector[1]))
    #problem.mu = param_vector[0] / (2.0 * (1.0 + param_vector[1]))
    problem.solve() 
    #mt("MCMC run")
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    if ndim == 1:
        return model_data[:,1]
    if ndim == 2:
        return model_data


#ForwardModelBase, Sensor objects, interface and response are mandatory.
import math
ProbeyeProblem = InverseProblem("My Problem")

ProbeyeProblem.add_parameter(name = "E", 
                            tex=r"$YoungsModulus$", 
                            info="Young's Modulus",
                            domain="[0, +oo)",
                            prior = Uniform(low=50.0, high=80)) # Uniform(low=50.0, high=150)

ProbeyeProblem.add_parameter(name = "nu", 
                            tex=r"$PoissonsRatio$", 
                            info="Poisson's Ratio",
                            domain="(0, 0.5)",
                            prior = Uniform(low=0.17, high=0.23)) #Normal(mean=0.15, std=0.1)math.log(0.15), std=2.3 Uniform(low=0, high=0.45)(mean=0.15, std=0.1)

#ProbeyeProblem.add_parameter(name = "sigma model",
#                            #domain="(0, +oo)",
#                            tex=r"$\sigma model$",
#                            info="Standard deviation, of zero-mean Gaussian noise model",
#                            prior=Uniform(low=0.0, high=0.8))

ProbeyeProblem.add_parameter(name = "sigma_model",
                            #domain="(0, +oo)",
                            #tex=r"$\sigma model$",
                            info="Standard deviation, of zero-mean Gaussian noise model",
                            value=0.0)

ProbeyeProblem.add_parameter(name = "sigma_meas",
                            #domain="(0, +oo)",
                            tex=r"$\sigma meas$",
                            info="Measurement error",
                            prior= Normal(mean=0, std=0.04))

ProbeyeProblem.add_experiment(name="Test1",
                            sensor_data={
                                "hor_disp": displacement_measured[:,0],
                                "ver_disp": displacement_measured[:,1],
                                "dummy": np.zeros((10,))
                            })

class FwdModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["E", "nu"]   #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = Sensor("dummy") #sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("hor_disp", std_model="sigma_model"),
                                Sensor("ver_disp", std_model="sigma_model")]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        #x = inp["x"] Don't need it as weight is already given in equations
        m = inp["E"]
        b = inp["nu"]
        displacement_results = forward_model_run(m, b)
        return {"hor_disp" : displacement_results[:,0],
                "ver_disp": displacement_results[:,1]}

ProbeyeProblem.add_forward_model(FwdModel("CantileverModel"), experiments=["Test1"])

ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="Test1",
    model_error="additive",
    measurement_error="sigma_meas")
)

emcee_solver = EmceeSolver(ProbeyeProblem)
inference_data = emcee_solver.run(n_steps=3000, n_initial_steps=1000,n_walkers=6)

true_values = {"E": 100, "nu": 0.2}

# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    emcee_solver.problem,
    true_values=true_values,
    focus_on_posterior=True,
    show_legends=True,
    title="Sampling results from emcee-Solver (pair plot)",
)

trace_plot_array = create_trace_plot(
    inference_data,
    emcee_solver.problem,
    title="Sampling results from emcee-Solver (trace plot)",
)