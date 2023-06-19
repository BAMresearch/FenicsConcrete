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
test1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1)

test1_x_component = add_noise_to_data(test1_disp[:,0], test1_total_sensors)
test1_y_component = add_noise_to_data(test1_disp[:,1], test1_total_sensors)
test1_disp = np.vstack((test1_x_component, test1_y_component)).T.flatten()


#Dense data (without sensors)s
#test1_disp = run_test(experiment, problem, 'left', [1e3, 0], 0) #np.copy is removed
#test2_disp = run_test(experiment, problem, 'bottom', [0,1e3], 0)
##tests1_disp = np.copy(run_test(experiment, problem, 'bottom', [1e3,0]))


list_of_disp = [test1_disp] #, tests1_disp
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

def forward_model_run(parameters):
    # Function to run the forward model

    #problem.E_m.value = parameters[0]*E_scaler #500e6
    #problem.E_d.value = parameters[1]*E_scaler
    #problem.nu_12.value = parameters[2]
    #problem.G_12.value =  parameters[3]*G_12_scaler + (parameters[0]*E_scaler)/(2*(1+parameters[2])) #parameters[3]*G_12_scaler 
    #problem.k_x.value =  10**(12 - (12-6)*parameters[4])  #1e15 - (1e15-1e5)*parameters[0] 
    #problem.k_y.value =  10**(12 - (12-6)*parameters[5])  #parameters[3]*G_12_scaler

    problem.k_x.value =  parameters[0] 
    problem.k_y.value =  parameters[1] 
    
    #Dense data (without sensors)
    #trial1_disp = run_test(experiment, problem, 'left', [1e3, 0], 0) #np.copy is removed
    #trial2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 0)

    #trialx_disp = np.reshape(run_test(experiment, problem, 'left', [1e3, 0], 0), (-1,2), order='C') #np.copy is removed
    #trialy_disp = np.reshape(run_test(experiment, problem, 'bottom', [0, 1e3], 0), (-1,2), order='C')
    #return combine_test_results([trialx_disp.flatten('F'), trialy_disp.flatten('F')]) #, trials1_disp

    #Sparse data (with sensors)
    _ = add_sensor(problem, 'left', test1_sensors_per_edge)
    trial1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1).flatten()
    return combine_test_results([trial1_disp])

from numpy import linalg as LA
cost_function_values = []
total_model_error = []
displacement_model_error = []
#parameter_values = {'E_m':[], 'E_d':[]}
#parameter_values['E_m'].append(param[0])
#parameter_values['E_d'].append(param[1])
#sparsity_factor = 1e-6
def cost_function(param, sparsity_factor=0):
    # Function to calculate the cost function
    displacement_model = forward_model_run(param)  
    #print(sparsity_factor)
    #delta_displacement = displacement_model - displacement_data
    delta_displacement = (displacement_model - displacement_data)/(displacement_data + 1e-10)
    #print('Inferred Parameters',param)
    function_evaluation = np.dot(delta_displacement, delta_displacement) 
    #cost_function_value = function_evaluation + sparsity_factor*LA.norm(param[np.array([1, 2, 3, 4, 5])], ord=1)
    displacement_model_error.append(function_evaluation)
    #total_model_error.append(cost_function_value)
    #return cost_function_value


import matplotlib.pyplot as plt
from matplotlib import cm


# Plotting the tendenacy of the parameters to tend to zero.

test_param = np.zeros((10,2))
test_param[:,0] = np.linspace(250e6,1000e6,10)
test_param[:,1] = np.linspace(250e6,1000e6,10)

for i in range(10):
    cost_function(test_param[i,:])



# Plotting the model error
import plotly.graph_objects as go
iteration_no = np.arange(1, len(displacement_model_error)+1)
fig = go.Figure()
#fig.add_trace(go.Scatter(x=iteration_no, y=total_model_error,  
#                    mode='lines+markers',
#                    name='Total Model Error'))
fig.add_trace(go.Scatter(x=test_param[:,1]/1e6, y=displacement_model_error,  
                    mode='lines+markers',
                    name='Displacement Model Error'))
#fig.update_layout(yaxis_type = "log")
fig.update_layout(title="Squared Relative Error in Displacements Vs. Spring Stiffness",
    xaxis_title="Spring Stiffness",
    yaxis_title="Squared Relative Error in Displacements")
fig.show() 
fig.write_html('Squared Relative Error in Displacements Vs. Spring Stiffness'+'.html')





































"""
#Isotropic data, istropic identification of tensile test
start_point = np.array([50e9, 0.15]) #, 1e8, 1e8
parameter_bounds = [(0, np.inf), (0, 0.45)] #, (0, np.inf), (0, np.inf)
res = minimize(cost_function, start_point, method='Nelder-Mead', bounds=parameter_bounds,#0.50.5
              options={'disp': True, 'maxiter':400}) 
print(res.x) 
print(p['G_12'])

E_m_trials = np.array(parameter_values['E_m'])
E_d_trials = np.array(parameter_values['E_d'])
cost_function_trials = np.array(cost_function_values) 

"""


