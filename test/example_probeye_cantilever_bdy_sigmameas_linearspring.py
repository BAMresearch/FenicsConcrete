
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
para['problem'] = 'cantilever_beam' #'tensile_test' #

# N/m², m, kg, sec, N
para['rho'] = 7750
para['g'] = 9.81
para['E'] = 210e9
para['length'] = 1
para['breadth'] = 0.2
para['load'] = 100
para['k_x'] = 1e14
para['k_y'] = 1e12


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
displacement_data = collect_sensor_solutions(problem.sensors, number_of_sensors)

#Clean the sensor data for the next simulation run
problem.clean_sensor_data()


def max_min_calculator(vec):
    return np.amax(np.absolute(vec)), np.amin(np.absolute(vec))

max_disp_hor_rest, min_disp_hor_rest = max_min_calculator(displacement_data[:20,0])
max_disp_ver_rest , min_disp_ver_rest = max_min_calculator(displacement_data[:20,1])

max_disp_hor_clamp, min_disp_hor_clamp = max_min_calculator(displacement_data[20:,0])
max_disp_ver_clamp , min_disp_ver_clamp = max_min_calculator(displacement_data[20:,1])


#max_disp_value_ver = np.amax(np.absolute(displacement_data[:,1]))
#max_disp_value_hor = np.amax(np.absolute(displacement_data[:,0]))
#
#min_disp_value_ver = np.amin(np.absolute(displacement_data[:,1]))
#min_disp_value_hor = np.amin(np.absolute(displacement_data[:,0]))
#
#sigma_error_hor = 0.1*max_disp_value_hor
#sigma_error_ver = 0.1*max_disp_value_ver

#np.random.seed(42) 
#distortion_hor = np.random.normal(0, 1e-11, (number_of_sensors)) #0.005
#distortion_ver = np.random.normal(0, 1e-11, (number_of_sensors)) 
#
#displacement_measured_hor = displacement_data[:,0] #+ distortion_hor
#displacement_measured_ver = displacement_data[:,1] #+ distortion_ver
#displacement_measured = np.stack((displacement_measured_hor, displacement_measured_ver), axis = -1)#.flatten()

distortion_hor_rest = np.random.normal(0, 1e-9, (20))
distortion_ver_rest = np.random.normal(0, 1e-9, (20))
distortion_hor_clamp = np.random.normal(0, 1e-11, (9))
distortion_ver_clamp = np.random.normal(0, 1e-10, (9))

displacement_hor_x_rest = displacement_data[:20,0]+distortion_hor_rest
displacement_ver_x_rest = displacement_data[:20,1]+distortion_ver_rest

displacement_hor_x_clamp = displacement_data[20:,0]+distortion_hor_clamp
displacement_ver_x_clamp = displacement_data[20:,1]+distortion_ver_clamp


def forward_model_run(param1, param2, param3, param4, ndim=2):
    problem.lambda_.value = param1 * param2 / ((1.0 + param2) * (1.0 - 2.0 * param2))
    problem.mu.value = param1 / (2.0 * (1.0 + param2))
    problem.k_x.value = param3
    problem.k_y.value = param4 #para['k_y']
    problem.solve() 
    #mt("MCMC run")
    model_data = collect_sensor_solutions(problem.sensors, number_of_sensors)
    problem.clean_sensor_data()
    if ndim == 1:
        return model_data[:,1]
    if ndim == 2:
        return model_data
#dataaa=forward_model_run(210e9, 0.28, 1e5, ndim=2) #k_x and k_y values aren't updating.
#dataaat=forward_model_run(210e9, 0.28, 1e10, ndim=2)

#ForwardModelBase, Sensor objects, interface and response are mandatory.
import math
ProbeyeProblem = InverseProblem("My Problem")

ProbeyeProblem.add_parameter(name = "E", 
                            tex=r"$YoungsModulus$", 
                            info="Young's Modulus",
                            #domain="[0, +oo)",
                            prior = LogNormal(mean=float(np.log(200*10**9))-0.5*0.1**2, std=0.1)) # Normal(mean=200*10**9, std=25*10**9)


ProbeyeProblem.add_parameter(name = "nu", 
                            tex=r"$PoissonsRatio$", 
                            info="Poisson's Ratio",
                            #domain="(0, 0.5)",
                            prior = LogNormal(mean=float(np.log(0.24))-0.5*0.15**2, std=0.15)) #Uniform(low=0.01, high=0.5)

ProbeyeProblem.add_parameter(name = "k_x",                     
                            tex=r"$K_x$",
                            info="Spring Stiffness in horizontal direction",
                            #domain="(0, +oo)",
                            prior=Uniform(low=1e12, high=1e15))

ProbeyeProblem.add_parameter(name = "k_y",                     
                            tex=r"$K_y$",
                            info="Spring Stiffness in vertical direction",
                            #domain="(0, +oo)",
                            prior=Uniform(low=1e11, high=1e13))

ProbeyeProblem.add_parameter(name = "sigma_model",
                            #domain="(0, +oo)",
                            #tex=r"$\sigma model$",
                            info="Standard deviation, of zero-mean Gaussian noise model",
                            value=0.0)

ProbeyeProblem.add_parameter(name = "sigma_x_rest",
                            #domain="(0, +oo)",
                            tex=r"$\sigma_{rest}x$",
                            info="Measurement error in  x rest",
                            prior= Normal(mean=1e-9, std=1e-8)) #0.004

ProbeyeProblem.add_parameter(name = "sigma_y_rest",
                            #domain="(0, +oo)",
                            tex=r"$\sigma_{rest}y$",
                            info="Measurement error y rest",
                            prior= Normal(mean=1e-9, std=1e-8))

ProbeyeProblem.add_parameter(name = "sigma_x_clamp",
                            #domain="(0, +oo)",
                            tex=r"$\sigma_{clamp}x$",
                            info="Measurement error x clamp",
                            prior= Normal(mean=1e-11, std=1e-10))

ProbeyeProblem.add_parameter(name = "sigma_y_clamp",
                            #domain="(0, +oo)",
                            tex=r"$\sigma_{clamp}y$",
                            info="Measurement error y clamp",
                            prior= Normal(mean=1e-10, std=1e-9))

ProbeyeProblem.add_experiment(name="Test1",
                            sensor_data={
                                "hor_disp_rest": displacement_hor_x_rest,
                                "dummy": np.zeros((10,))
                            })

ProbeyeProblem.add_experiment(name="Test2",
                            sensor_data={
                                "ver_disp_rest": displacement_ver_x_rest, "dummy": np.zeros((10,))       
                            })

ProbeyeProblem.add_experiment(name="Test3",
                            sensor_data={
                                "hor_disp_clamp": displacement_hor_x_clamp, "dummy": np.zeros((10,))                              
                            })

ProbeyeProblem.add_experiment(name="Test4",
                            sensor_data={
                                "ver_disp_clamp": displacement_ver_x_clamp, "dummy": np.zeros((10,))                            
                            })

class FwdModel1(ForwardModelBase):
    def interface(self):
        self.parameters = ["E", "nu", "k_x", "k_y"]   #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = Sensor("dummy") #sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("hor_disp_rest", std_model="sigma_model",)]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        #x = inp["x"] Don't need it as weight is already given in equations
        m = inp["E"]
        b = inp["nu"]
        t = inp["k_x"]
        u = inp["k_y"]
        displacement_results = forward_model_run(m, b, t, u)
        return {"hor_disp_rest" : displacement_results[:20,0]}



class FwdModel2(ForwardModelBase):
    def interface(self):
        self.parameters = ["E", "nu", "k_x", "k_y"]   #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = Sensor("dummy") #sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("ver_disp_rest", std_model="sigma_model")]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        #x = inp["x"] Don't need it as weight is already given in equations
        m = inp["E"]
        b = inp["nu"]
        t = inp["k_x"]
        u = inp["k_y"]
        displacement_results = forward_model_run(m, b, t, u)
        return {"ver_disp_rest": displacement_results[:20,1]}



class FwdModel3(ForwardModelBase):
    def interface(self):
        self.parameters = ["E", "nu", "k_x", "k_y"]   #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = Sensor("dummy") #sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("hor_disp_clamp", std_model="sigma_model")]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        #x = inp["x"] Don't need it as weight is already given in equations
        m = inp["E"]
        b = inp["nu"]
        t = inp["k_x"]
        u = inp["k_y"]
        displacement_results = forward_model_run(m, b, t, u)
        return {"hor_disp_clamp": displacement_results[20:,0]}


class FwdModel4(ForwardModelBase):
    def interface(self):
        self.parameters = ["E", "nu", "k_x", "k_y"]   #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = Sensor("dummy") #sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("ver_disp_clamp", std_model="sigma_model")]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        #x = inp["x"] Don't need it as weight is already given in equations
        m = inp["E"]
        b = inp["nu"]
        t = inp["k_x"]
        u = inp["k_y"]
        displacement_results = forward_model_run(m, b, t, u)
        return {"ver_disp_clamp": displacement_results[20:,1]}


ProbeyeProblem.add_forward_model(FwdModel1("CantileverModel1"), experiments=["Test1"])
ProbeyeProblem.add_forward_model(FwdModel2("CantileverModel2"), experiments=["Test2"])
ProbeyeProblem.add_forward_model(FwdModel3("CantileverModel3"), experiments=["Test3"])
ProbeyeProblem.add_forward_model(FwdModel4("CantileverModel4"), experiments=["Test4"])

ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="Test1",
    model_error="additive",
    measurement_error="sigma_x_rest")
)

ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="Test2",
    model_error="additive",
    measurement_error="sigma_y_rest")
)

ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="Test3",
    model_error="additive",
    measurement_error="sigma_x_clamp")
)

ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="Test4",
    model_error="additive",
    measurement_error="sigma_y_clamp")
)

emcee_solver = EmceeSolver(ProbeyeProblem)
inference_data = emcee_solver.run(n_steps=1500, n_initial_steps=150,n_walkers=20)

true_values = {"E": para['E'], "nu": para['nu'], "k_x":para['k_x'], "k_y":para['k_y']} 

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