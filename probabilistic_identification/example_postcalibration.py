
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

with open('probabilistic_identification/test_config.json', 'r') as f: 
    json_object = json.loads(f.read()) 

parameters_list = ["E", "nu", "sigma"] #"E_d",
measurement_data = np.loadtxt(json_object.get('Data').get('measurement_data'), delimiter=' ')
posterior_data = np.loadtxt(json_object.get('MCMC').get('chain_name'), delimiter=',')
posterior_data = posterior_data.reshape(posterior_data.shape[0], posterior_data.shape[1]// len(parameters_list), len(parameters_list))
posterior_data = np.transpose(posterior_data, (1,0,2))

# Adding sensors to the problem definition.
def add_sensor(_problem, _dirichlet_bdy, _sensors_num_edge_hor, _sensors_num_edge_ver): 
    sensor = []
    if _dirichlet_bdy == 0: #'left'
        for i in range(_sensors_num_edge_hor): 
            #print((p['length']*(i+1))/_sensors_num_edge_hor)
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[(p['length']*(i+1))/_sensors_num_edge_hor, 0, 0]]))) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[(p['length']*(i+1))/_sensors_num_edge_hor, p['breadth'], 0]])))
        
        for i in range(_sensors_num_edge_ver):
            #print((p['breadth']*(i+1))/(_sensors_num_edge_ver+1))
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length'], (p['breadth']*(i+1))/(_sensors_num_edge_ver+1), 0]])))

        for i in range(len(sensor)):
            _problem.add_sensor(sensor[i])
        return len(sensor)

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



test1_data = run_test(experiment, problem, 0, [1e3, 0], 1)



chain_total = 2
chain_length = 2000
likelihood_predictive = np.zeros((chain_total,chain_length,n_tests))

x_plot = np.zeros((chain_total, chain_length, n_tests))

for chain_index in range(chain_total): #chain index
    for k in range(n_tests): #data point index n_tests
        x = x_test[k]
        y = y_test[k]
        for chain_draw in range(chain_length): #step number of chain/inferred parameters

            problem.E.value = inp["E"] 
            problem.nu.value = inp["nu"]
            test1_data = run_test(experiment, problem, 0, [1e3, 0], 1)
            a = posterior[chain_draw,chain_index,0]
            b = posterior[chain_draw,chain_index,1]
            sigma = posterior[chain_draw,chain_index,2]
            likelihood_predictive[chain_index,chain_draw,k] = stats.norm.pdf(y,a*x+b,sigma)   #a*x+b + np.random.normal(loc=0, scale=sigma)

posterior_predictive = np.mean(np.mean(likelihood_predictive, axis=1), axis=0)
#print(posterior_predictive)

plt.plot(x_test, posterior_predictive, "o", label="posterior predictive")
plt.title("Posterior Predictive vs. Observations")
plt.xlabel("Observed Data")
plt.ylabel("Posterior Predictive")
plt.legend()
plt.show()




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


#######################################################################################################################################################
#######################################################################################################################################################
#3rd Step - Post Processing
#######################################################################################################################################################
#######################################################################################################################################################
