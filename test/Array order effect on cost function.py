import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import matplotlib.pyplot as matplot
import math

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

p['constitutive'] = 'isotropic'
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
    if dirichlet_bdy == 'left':
        for i in range(10): #20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/sensors_per_side*(i+1), 0, 0]]))) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/sensors_per_side*(i+1), p['breadth'], 0]])))

        for i in range(len(sensor)):
            prob.add_sensor(sensor[i])
        return len(sensor)

    elif dirichlet_bdy == 'bottom':
        for i in range(5): #20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[0, p['breadth']/sensors_per_side*(i+1), 0]]))) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length'], p['breadth']/sensors_per_side*(i+1), 0]])))

        for i in range(len(sensor)):
            prob.add_sensor(sensor[i])
        return len(sensor)


def run_test(exp, prob, dirichlet_bdy, load, sensor_flag = 0):
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
    return clean_data #+ np.random.normal(0, 0.01 * min_disp, no_of_sensors)

#Sparse data (with sensors)
test1_sensors_per_edge = 10
test1_total_sensors = add_sensor(problem, 'left', test1_sensors_per_edge)
test1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1).flatten()

#test1_x_component = add_noise_to_data(test1_disp[:,0], test1_total_sensors)
#test1_y_component = add_noise_to_data(test1_disp[:,1], test1_total_sensors)
#test1_disp = np.vstack((test1_x_component, test1_y_component)).T.flatten()

test2_sensors_per_edge = 5
test2_total_sensors = add_sensor(problem, 'bottom', test2_sensors_per_edge)
test2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 1).flatten()


#Sparse data (with sensors)
test3_sensors_per_edge = 10
test3_total_sensors = add_sensor(problem, 'left', test3_sensors_per_edge)
test3_disp = run_test(experiment, problem, 'left', [1e3, 0], 1).flatten('F')


test4_sensors_per_edge = 5
test4_total_sensors = add_sensor(problem, 'bottom', test4_sensors_per_edge)
test4_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 1).flatten('F')

#test2_x_component = add_noise_to_data(test2_disp[:,0], test2_total_sensors)
#test2_y_component = add_noise_to_data(test2_disp[:,1], test2_total_sensors)
#test2_disp = np.vstack((test2_x_component, test2_y_component)).T.flatten()

#Dense data (without sensors)s
#test1_disp = run_test(experiment, problem, 'left', [1e3, 0], 0) #np.copy is removed
#test2_disp = run_test(experiment, problem, 'bottom', [0,1e3], 0)
##tests1_disp = np.copy(run_test(experiment, problem, 'bottom', [1e3,0]))

# Not in Use
#test1_disp = np.reshape(run_test(experiment, problem, 'left', [1e3, 0], 0), (-1,2), order = 'C') #np.copy is removed
#test2_disp = np.reshape(run_test(experiment, problem, 'bottom', [0,1e3], 0), (-1,2), order='C')
#list_of_disp = [test1_disp.flatten('F'), test2_disp.flatten('F')] #, tests1_disp

list_of_disp = [test1_disp] #, tests1_disp , test2_disp
#num_of_tests = str(len(list_of_disp)) + ' tests' 
displacement_data_1 = combine_test_results(list_of_disp)  

list_of_disp_ = [test3_disp] #, tests1_disp , test4_disp
#num_of_tests = str(len(list_of_disp)) + ' tests' 
displacement_data_2 = combine_test_results(list_of_disp_) 

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
p['uncertainties'] = [0,2]
p['E_m'] = 210e6
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  210e6/(2*(1+0.28)) #(0.5*1e5)/(1+0.3)
p['k_x'] = 1e12
p['k_y'] = 1e12


experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.


E_scaler = 500e6
G_12_scaler = 250e6 
#K_scaler = 1e2 #1e7

def forward_model_run(parameters, keyword):
    # Function to run the forward model

    problem.E_m.value = parameters[0]*E_scaler #500e6
    problem.E_d.value = parameters[1]*E_scaler
    problem.nu_12.value = parameters[2]
    problem.G_12.value =  parameters[3]*G_12_scaler + (parameters[0]*E_scaler)/(2*(1+parameters[2])) #(parameters[3] + (parameters[0])/(2*(1+parameters[2])))*G_12_scaler 
    problem.k_x.value =  10**(12 - (12-6)*parameters[4])  #1e15 - (1e15-1e5)*parameters[0] 
    problem.k_y.value =  10**(12 - (12-6)*parameters[5])  #parameters[3]*G_12_scaler

    #Sparse data (with sensors)
    if keyword == 'XXYY':
        _ = add_sensor(problem, 'left', test1_sensors_per_edge)
        trial1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1).flatten('F')
        #_ = add_sensor(problem, 'bottom', test2_sensors_per_edge)
        #trial2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 1).flatten('F')
    else:
        _ = add_sensor(problem, 'left', test1_sensors_per_edge)
        trial1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1).flatten()
        #_ = add_sensor(problem, 'bottom', test2_sensors_per_edge)
        #trial2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 1).flatten()
    return combine_test_results([trial1_disp]) #, trial2_disp

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.16f}'.format})
from numpy import linalg as LA
cost_function_values = []
total_model_error = []
displacement_model_error = []
sparsity_factor = 1e-7
def cost_function(param):
    # Function to calculate the cost function
    displacement_model_1 = forward_model_run(param, 'XYXY')  
    delta_displacement_1 = displacement_model_1 - displacement_data_1

    displacement_model_2 = forward_model_run(param, 'XXYY')  
    delta_displacement_2 = displacement_model_2 - displacement_data_2

    a1 = np.vstack((delta_displacement_2[:20], delta_displacement_2[20:40])).T.flatten()
    a2 = np.vstack((delta_displacement_2[40:50], delta_displacement_2[50:60])).T.flatten()
    delta_displacement_2_new = combine_test_results([a1, a2])

    function_evaluation_1 = np.dot(delta_displacement_1, delta_displacement_1) 
    function_evaluation_2 = np.dot(delta_displacement_2, delta_displacement_2)
    function_evaluation_3 = np.dot(delta_displacement_2_new, delta_displacement_2_new)
    print(function_evaluation_1 - function_evaluation_2)
    print(function_evaluation_1 - function_evaluation_3)


a = cost_function([0.42, 1e-5, 0.28, 1e-4, 1e-1, 1e-1])
print(a)



