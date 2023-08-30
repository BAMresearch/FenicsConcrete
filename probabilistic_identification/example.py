
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
import pandas as pd
#from scipy import optimize


with open('probabilistic_identification/test_config.json', 'r') as f: 
    json_object = json.loads(f.read()) 

# Adding sensors to the problem definition.
def add_sensor(_problem, _dirichlet_bdy, _sensors_num_edge_hor, _sensors_num_edge_ver): 
    sensor = []
    if _dirichlet_bdy == 0: #'left'
        for i in range(_sensors_num_edge_hor): 
            #print((p['length']*(i+1))/_sensors_num_edge_hor)
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
    with open('sensor_data.json', 'w') as f:
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
    store_sensor_data(prob)
    if sensor_flag == 0:
        return prob.displacement.x.array
    elif sensor_flag == 1 :
        counter=0
        displacement_at_sensors = np.zeros((len(prob.sensors),2))
        for i in prob.sensors:
            displacement_at_sensors[counter] = prob.sensors[i].data[-1]
            counter += 1
        prob.sensors = fenicsX_concrete.sensors.Sensors()
        return displacement_at_sensors#.flatten()
    
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
#1st Step - Data Generation
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
p['uncertainties'] = [0,2]
p['k_x'] = 1e8
p['k_y'] = 1e8

p['constitutive'] = 'isotropic' #'orthotropic' 
p['nu'] = 0.28 

# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² 

p['dirichlet_bdy'] = 'left'

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

#Sparse data (with sensors)
sensors_num_edge_hor = 10
sensors_num_edge_ver = 4
test1_sensors_total_num = add_sensor(problem, 0, sensors_num_edge_hor, sensors_num_edge_ver)

sensor_positions = np.zeros((test1_sensors_total_num, 3))
counter = 0
for i in problem.sensors:
    sensor_positions[counter] = problem.sensors[i].where[0]
    counter += 1

test1_data = run_test(experiment, problem, 0, [1e3, 0], 1)
test1_x_component = add_noise_to_data(test1_data[:,0], test1_sensors_total_num)
test1_y_component = add_noise_to_data(test1_data[:,1], test1_sensors_total_num)
test1_data = np.vstack((test1_x_component, test1_y_component)).T.flatten()


# Not in Use
#test1_disp = np.reshape(run_test(experiment, problem, 'left', [1e3, 0], 0), (-1,2), order = 'C') #np.copy is removed
#test2_disp = np.reshape(run_test(experiment, problem, 'bottom', [0,1e3], 0), (-1,2), order='C')
#list_of_disp = [test1_disp.flatten('F'), test2_disp.flatten('F')] #, tests1_disp

#list_of_disp = [test1_disp, test2_disp] #, tests1_disp
#num_of_tests = str(len(list_of_disp)) + ' tests' 
displacement_data = test1_data # combine_test_results(list_of_disp)  



subdict = {   
    "data" : problem.sensors['DisplacementSensor'].data,
    "where" : problem.sensors['DisplacementSensor'].where,
    "alphabetical_position" : problem.sensors['DisplacementSensor'].alphabetical_position}

mydict = {
    "sensor1" : subdict}

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
                ],
    "MCMC": {
            "parameter_scaling" : True,
            "nburn": 125,
            "nsteps": 125,
            "pair_plot_name": "pair_plot_scaled_parameters.png",
            "trace_plot_name": "trace_plot_scaled_parameters.png"
          }
        
    }   """
            
mydict = {
    "sensor1" : subdict}
json_string = json.dumps(mydict , indent = 3)
with open('sensor_data.json', 'w') as f:
    f.write(json_string) 










pd.DataFrame(displacement_data).to_csv(json_object.get('Data').get('measurement_data'), index=False, header=False)
pd.DataFrame(sensor_positions).to_csv(json_object.get('Data').get('sensor_positions'), index=False, header=False)
#############################################################################################################################
#############################################################################################################################
#2nd Step - Inverse Problem
#############################################################################################################################
#############################################################################################################################


# Kgmms⁻2/mm², mm, kg, sec, N
#p['constitutive'] = 'isotropic' #'orthotropic'
p['uncertainties'] = [0] #,2
#p['E'] = 210e6
#p['nu'] = 0.28 #0.3

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

#ForwardModelBase, Sensor objects, interface and response are mandatory.
ProbeyeProblem = InverseProblem("My Problem")


def prior_func_selection(para :list):
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


#Select the parameters for inference from the json file.
parameters_list = ["E", "nu", "sigma"] #"E_d",
for parameter in json_object.get('parameters'):
    if parameter['name'] in parameters_list:
        ProbeyeProblem.add_parameter(name = parameter['name'], 
                                     tex =  parameter['tex'],
                                     info = parameter['info'], 
                                     domain = parameter['domain'] if parameter['domain'] != None else "(-oo, +oo)",
                                     prior = prior_func_selection(parameter['prior']))  

ProbeyeProblem.add_experiment(name="tensile_test_1",
                            sensor_data={
                                "disp": test1_data,
                                "dirichlet_bdy": 0,  #Provided must be a 1D array.
                                "neumann_bdy": [1000, 0],
                                "sensors_per_edge" : 10
                            })


class FEMModel(ForwardModelBase):
    def interface(self):
        self.parameters = parameters_list  # "G_12", "k_x", "k_y",  #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = [Sensor("dirichlet_bdy"), Sensor("neumann_bdy"), Sensor("sensors_per_edge")]#sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("disp", std_model="sigma")]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        #if inp["E_m"] < inp["E_d"]:
        #    model_output = np.ones(inp["sensors_per_edge"]*4)*1e20#np.inf
        #    return {"disp": model_output}
        #else:
        #if json_object.get('MCMC').get('parameter_scaling') == True:
            #problem.E_1.value = (inp["E_d"] + inp["E_2"])*500*10**6   #0.5*(inp["E_1"]-inp["E_2"])*500*10**6    
            #problem.E_2.value = inp["E_2"]*500*10**6    #0.5*(inp["E_1"]+inp["E_2"])*500*10**6   
            #problem.nu_12.value = inp["nu"]
            #problem.G_12.value = inp["G_12"]*250*10**6 + (inp["E_2"]*500*10**6 )/(2*(1+inp["nu"]))
            #problem.k_x.value =  (2000-2000*inp["k_x"])*10**6   #10**(12-6*inp["k_x"]) #inp["k_x"]  
            #problem.k_y.value =  (2000-2000*inp["k_y"])*10**6 #10**(12-6*inp["k_y"]) #inp["k_y"] #
        #else:
            #problem.G_12.value = 82.03125*10**6 # inp["G_12"]#*250*10**6 # + (inp["E_m"] )/(2*(1+inp["nu"])) 82.03125*10**6 # 
            #problem.k_x.value =  (2000-2000*inp["k_x"])*10**6   #10**(12-6*inp["k_x"]) #inp["k_x"]  
            #problem.k_y.value =  (2000-2000*inp["k_y"])*10**6 #10**(12-6*inp["k_y"]) #inp["k_y"] #
        problem.E.value = inp["E"] 
        problem.nu.value = inp["nu"]
        dirichlet_bdy = inp["dirichlet_bdy"]
        neumann_bdy = inp["neumann_bdy"]
        sensors_per_edge = inp["sensors_per_edge"]
        _ = add_sensor(problem, dirichlet_bdy, sensors_num_edge_hor, sensors_num_edge_ver)
        model_output = run_test(experiment, problem, dirichlet_bdy, neumann_bdy, 1).flatten()
        return {"disp" : model_output}


ProbeyeProblem.add_forward_model(FEMModel("LinearElasticMaterial"), experiments=["tensile_test_1"])


ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="tensile_test_1",
    model_error="additive",
    ) #measurement_error="sigma_x_rest"
)

emcee_solver = EmceeSolver(ProbeyeProblem)
inference_data = emcee_solver.run(n_steps=json_object.get('MCMC').get('nsteps'), n_initial_steps=json_object.get('MCMC').get('nburn')) #,n_walkers=20

#######################################################################################################################################################
#######################################################################################################################################################
#3rd Step - Post Processing
#######################################################################################################################################################
#######################################################################################################################################################

# Saving Arviz Data to json.
inference_data.to_json(json_object.get('MCMC').get('arviz_data_name')) 

# Saving the posterior as a csv file
posterior = emcee_solver.raw_results.get_chain()
np.savetxt(json_object.get('MCMC').get('chain_name'), posterior.reshape(posterior.shape[0], -1), delimiter=",")

#import emcee
#emcee.autocorr.integrated_time(emcee_solver.raw_results.get_chain())

#true_values = {"E_m": 210*10**6, "E_d": 0., "nu": 0.28} #"G_12": 82.03125*10**6
#true_values = {"E_m": 0.42, "E_d": 0., "nu": 0.28, "G_12": 0.328} # , "G_12": 82.03125*10**6 , "k_x":3*10**9, "k_y":10**11
#if json_object.get('MCMC').get('parameter_scaling') == True:
#    true_values = {"E_2": 0.42, "E_d": 0., "nu": 0.28, "G_12": 0.} #"E_d": 0., 
#else:
true_values = {"E": 210*10**6, "nu": 0.28,} #"E_d": 0., 


# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    emcee_solver.problem,
    true_values=true_values,
    focus_on_posterior=True,
    show_legends=True,
    title="Sampling results from emcee-Solver (pair plot)",
    show=False
)
fig1 = pair_plot_array.ravel()[0].figure
fig1.savefig(json_object.get('MCMC').get('pair_plot_name'))

trace_plot_array = create_trace_plot(
    inference_data,
    emcee_solver.problem,
    title="Sampling results from emcee-Solver (trace plot)",
    show=False
)
fig2 = trace_plot_array.ravel()[0].figure
fig2.savefig(json_object.get('MCMC').get('trace_plot_name'))
