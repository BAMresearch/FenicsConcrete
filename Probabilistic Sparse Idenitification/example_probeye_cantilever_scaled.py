
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
from scipy import optimize

#########################################################################
#########################################################################
#1st Step - Data Generation
#########################################################################
#########################################################################

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
p['nu'] = 0.28 

# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² 

p['dirichlet_bdy'] = 'left'

# Adding sensors to the problem definition.
def add_sensor(prob, dirichlet_bdy, sensors_per_side):
    sensor = []
    if dirichlet_bdy == 0: #'left'
        for i in range(10): 
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/sensors_per_side*(i+1), 0, 0]]))) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/sensors_per_side*(i+1), p['breadth'], 0]])))

        for i in range(len(sensor)):
            prob.add_sensor(sensor[i])
        return len(sensor)

    elif dirichlet_bdy == 1: #'bottom'
        for i in range(5): 
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[0, p['breadth']/sensors_per_side*(i+1), 0]]))) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length'], p['breadth']/sensors_per_side*(i+1), 0]])))

        for i in range(len(sensor)):
            prob.add_sensor(sensor[i])
        return len(sensor)


def run_test(exp, prob, dirichlet_bdy, load, sensor_flag = 0):
    if dirichlet_bdy == 0:
        dirichlet_bdy = 'left'
    elif dirichlet_bdy == 1:
        dirichlet_bdy = 'bottom'
    prob.p.dirichlet_bdy = dirichlet_bdy
    exp.p.dirichlet_bdy = dirichlet_bdy
    prob.p.load = load
    prob.experiment.bcs = prob.experiment.create_displ_bcs(prob.experiment.V)
    prob.apply_neumann_bc()
    prob.calculate_bilinear_form()
    prob.solve()
    #prob.pv_plot("Displacement.xdmf")
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

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

def add_noise_to_data(clean_data, no_of_sensors):
    max_disp = np.amax(np.absolute(clean_data))
    min_disp = np.amin(np.absolute(clean_data))
    print('Max', max_disp, 'Min', min_disp)
    return clean_data + np.random.normal(0, 0.01 * min_disp, no_of_sensors)

#Sparse data (with sensors)
test1_sensors_per_edge = 10
test1_total_sensors = add_sensor(problem, 0, test1_sensors_per_edge)
test1_disp = run_test(experiment, problem, 0, [1e3, 0], 1)

test1_x_component = add_noise_to_data(test1_disp[:,0], test1_total_sensors)
test1_y_component = add_noise_to_data(test1_disp[:,1], test1_total_sensors)
test1_disp = np.vstack((test1_x_component, test1_y_component)).T.flatten()

test2_sensors_per_edge = 5
test2_total_sensors = add_sensor(problem, 1, test2_sensors_per_edge)
test2_disp = run_test(experiment, problem, 1, [0, 1e3], 1)

test2_x_component = add_noise_to_data(test2_disp[:,0], test2_total_sensors)
test2_y_component = add_noise_to_data(test2_disp[:,1], test2_total_sensors)
test2_disp = np.vstack((test2_x_component, test2_y_component)).T.flatten()

#Dense data (without sensors)s
#test1_disp = run_test(experiment, problem, 'left', [1e3, 0], 0) #np.copy is removed
#test2_disp = run_test(experiment, problem, 'bottom', [0,1e3], 0)
##tests1_disp = np.copy(run_test(experiment, problem, 'bottom', [1e3,0]))

# Not in Use
#test1_disp = np.reshape(run_test(experiment, problem, 'left', [1e3, 0], 0), (-1,2), order = 'C') #np.copy is removed
#test2_disp = np.reshape(run_test(experiment, problem, 'bottom', [0,1e3], 0), (-1,2), order='C')
#list_of_disp = [test1_disp.flatten('F'), test2_disp.flatten('F')] #, tests1_disp

list_of_disp = [test1_disp, test2_disp] #, tests1_disp
#num_of_tests = str(len(list_of_disp)) + ' tests' 
displacement_data = combine_test_results(list_of_disp)  

#displacement_data.shape[0]
#0.001*displacement_data
#np.random.multivariate_normal(np.shape(displacement_data)[0], np.eye(2), 1)


#########################################################################
#########################################################################
#2nd Step - Inverse Problem
#########################################################################
#########################################################################

# Kgmms⁻2/mm², mm, kg, sec, N
p['constitutive'] = 'orthotropic'
p['uncertainties'] = [0, 2] #,2
p['E_m'] = 210e6
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  210e6/(2*(1+0.28)) #(0.5*1e5)/(1+0.3)
p['k_x'] = 1e12
p['k_y'] = 1e12


experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

#ForwardModelBase, Sensor objects, interface and response are mandatory.
import math
ProbeyeProblem = InverseProblem("My Problem")

ProbeyeProblem.add_parameter(name = "E_m", 
                            tex=r"$E_m$", 
                            info="Young's Modulus of the material",
                            domain="(0, +oo)",
                            prior = Uniform(low=0, high=1)) #  LogNormal(mean=float(np.log(200*10**6))-0.5*0.1**2, std=0.1)

""" ProbeyeProblem.add_parameter(name = "E_d", #*10**6
                            tex=r"$E_d$", 
                            info="Young's Modulus of the material",
                            domain="(0, +oo)",
                            prior = Uniform(low=0, high=500*10**6)) #Normal(mean=200*10**6, std=25*10**9) Uniform(low=0, high=400*10**6)

ProbeyeProblem.add_parameter(name = "nu", 
                            tex=r"$\nu$", 
                            info="Poisson's Ratio",
                            domain="[0, 0.45)",
                            prior = Uniform(low=0.01, high=0.45)) # LogNormal(mean=float(np.log(0.24))-0.5*0.15**2, std=0.15) """

""" ProbeyeProblem.add_parameter(name = "G_12", 
                            tex=r"$G_{12}$", 
                            info="Shear Modulus",
                            domain="(0, +oo)",
                            prior = Uniform(low=0, high=100*10**6)) # LogNormal(mean=float(np.log(0.24))-0.5*0.15**2, std=0.15) """

""" ProbeyeProblem.add_parameter(name = "k_x",       #100*10**6              
                            tex=r"$K_x$",
                            info="Spring Stiffness in horizontal direction",
                            domain="(0, 1)",
                            prior=Uniform(low=0, high=1)) #(low=10**6, high=10**12)) (low=0, high=1)

ProbeyeProblem.add_parameter(name = "k_y",                     
                            tex=r"$K_y$",
                            info="Spring Stiffness in vertical direction",
                            domain="(0, 1)",
                            prior=Uniform(low=0, high=1))  """


""" ProbeyeProblem.add_parameter(name = "k_x",       #100*10**6              
                            tex=r"$K_x$",
                            info="Spring Stiffness in horizontal direction",
                            domain="[0, 1]",
                            prior=Uniform(low=0, high=1))#(scale=1/(6*math.log(10)), shift=2)) #(low=10**6, high=10**12)) (low=0, high=1)

ProbeyeProblem.add_parameter(name = "k_y",                     
                            tex=r"$K_y$",
                            info="Spring Stiffness in vertical direction",
                            domain="[0, 1]",
                            prior=Uniform(low=0, high=1))#(scale=1/(6*math.log(10)), shift=2)) 

ProbeyeProblem.add_parameter(name = "sigma", 
                            tex=r"$\sigma_{model}$",
                            domain="(0, 1)",
                            info="Standard deviation, of zero-mean Gaussian noise model",
                            prior=Uniform(low=1e-6, high=1e-5),) """

""" ProbeyeProblem.add_parameter(name = "sigma_x_rest",
                            #domain="(0, +oo)",
                            tex=r"$\sigma_{rest}x$",
                            info="Measurement error in  x rest",
                            prior= Normal(mean=4e-5, std=1e-6)) #0.004

ProbeyeProblem.add_parameter(name = "sigma_y_rest",
                            #domain="(0, +oo)",
                            tex=r"$\sigma_{rest}y$",
                            info="Measurement error y rest",
                            prior= Normal(mean=12e-6, std=1e-7)) """

#ProbeyeProblem.add_parameter(name = "sigma_x_clamp",
#                            #domain="(0, +oo)",
#                            tex=r"$\sigma_{clamp}x$",
#                            info="Measurement error x clamp",
#                            prior= Normal(mean=1e-11, std=1e-10))
#
#ProbeyeProblem.add_parameter(name = "sigma_y_clamp",
#                            #domain="(0, +oo)",
#                            tex=r"$\sigma_{clamp}y$",
#                            info="Measurement error y clamp",
#                            prior= Normal(mean=1e-10, std=1e-9))

ProbeyeProblem.add_experiment(name="tensile_test_x",
                            sensor_data={
                                "disp": test1_disp,
                                "dirichlet_bdy": 0,  #Provided must be a 1D array.
                                "neumann_bdy": [1000, 0],
                                "sensors_per_edge" : 10
                            })

ProbeyeProblem.add_experiment(name="tensile_test_y",
                            sensor_data={
                                "disp": test2_disp,
                                "dirichlet_bdy": 1,
                                "neumann_bdy": [0, 1000],                                
                                "sensors_per_edge" : 5
                            })



class FEMModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["E_m", "E_d", "nu", "k_x", "k_y"]  # "G_12", "k_x", "k_y",  #E and nu must have been already defined beforehand using add_parameter. # three attributes are must here.
        self.input_sensors = [Sensor("dirichlet_bdy"), Sensor("neumann_bdy"), Sensor("sensors_per_edge")]#sensor provides a way for forward model to interact with experimental data.
        self.output_sensors = [Sensor("disp", std_model="sigma")]

    def response(self, inp: dict) -> dict:    #forward model evaluation
        problem.E_m.value = inp["E_m"]#*500*10**6   
        problem.E_d.value = inp["E_d"]#*500*10**6     
        problem.nu_12.value = inp["nu"]
        problem.G_12.value = 82.03125*10**6 # inp["G_12"]#*250*10**6 # + (inp["E_m"] )/(2*(1+inp["nu"])) 82.03125*10**6 # 
        problem.k_x.value =  (2000-2000*inp["k_x"])*10**6   #10**(12-6*inp["k_x"]) #inp["k_x"]  
        problem.k_y.value =  (2000-2000*inp["k_y"])*10**6 #10**(12-6*inp["k_y"]) #inp["k_y"] #

        dirichlet_bdy = inp["dirichlet_bdy"]
        neumann_bdy = inp["neumann_bdy"]
        sensors_per_edge = inp["sensors_per_edge"]

        _ = add_sensor(problem, dirichlet_bdy, sensors_per_edge)
        model_output = run_test(experiment, problem, dirichlet_bdy, neumann_bdy, 1).flatten()
        return {"disp" : model_output}



ProbeyeProblem.add_forward_model(FEMModel("LinearElasticOrthotropicMaterial"), experiments=["tensile_test_x", "tensile_test_y"])


ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="tensile_test_x",
    model_error="additive",
    ) #measurement_error="sigma_x_rest"
)

ProbeyeProblem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="tensile_test_y",
    model_error="additive",
    ) #measurement_error="sigma_y_rest"
)

emcee_solver = EmceeSolver(ProbeyeProblem)
burn_in = 400
after_burn_in = 900
inference_data = emcee_solver.run(n_steps=after_burn_in, n_initial_steps=burn_in) #,n_walkers=20

#true_values = {"E_m": 210*10**6, "E_d": 0., "nu": 0.28} #"G_12": 82.03125*10**6
#true_values = {"E_m": 0.42, "E_d": 0., "nu": 0.28, "G_12": 0.328} # , "G_12": 82.03125*10**6 , "k_x":3*10**9, "k_y":10**11
true_values = {"E_m": 0.42, "E_d": 0., "nu": 0.28}

# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    emcee_solver.problem,
    true_values=true_values,
    focus_on_posterior=True,
    show_legends=True,
    title="Sampling results from emcee-Solver (pair plot)",
)
fig = pair_plot_array.ravel()[0].figure
fig.savefig("pair_plot_scaled_G12_"+str(burn_in)+"_"+str(after_burn_in)+"_spring.png")

trace_plot_array = create_trace_plot(
    inference_data,
    emcee_solver.problem,
    title="Sampling results from emcee-Solver (trace plot)",
)
fig = trace_plot_array.ravel()[0].figure
fig.savefig("trace_plot_scaled_G12_"+str(burn_in)+"_"+str(after_burn_in)+"_spring.png")