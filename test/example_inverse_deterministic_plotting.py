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
    return clean_data + np.random.normal(0, 0.01 * min_disp, no_of_sensors)

#Sparse data (with sensors)
test1_sensors_per_edge = 10
test1_total_sensors = add_sensor(problem, 'left', test1_sensors_per_edge)
test1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1)

test1_x_component = add_noise_to_data(test1_disp[:,0], test1_total_sensors)
test1_y_component = add_noise_to_data(test1_disp[:,1], test1_total_sensors)
test1_disp = np.vstack((test1_x_component, test1_y_component)).T.flatten()

test2_sensors_per_edge = 5
test2_total_sensors = add_sensor(problem, 'bottom', test2_sensors_per_edge)
test2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 1)

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

    problem.E_m.value = parameters[0]*E_scaler #500e6
    problem.E_d.value = parameters[1]*E_scaler
    problem.nu_12.value = parameters[2]
    problem.G_12.value =  parameters[3]*G_12_scaler + (parameters[0]*E_scaler)/(2*(1+parameters[2])) #parameters[3]*G_12_scaler 
    problem.k_x.value =  10**(12 - (12-6)*parameters[4])  #1e15 - (1e15-1e5)*parameters[0] 
    problem.k_y.value =  10**(12 - (12-6)*parameters[5])  #parameters[3]*G_12_scaler
    
    #Dense data (without sensors)
    #trial1_disp = run_test(experiment, problem, 'left', [1e3, 0], 0) #np.copy is removed
    #trial2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 0)

    #trialx_disp = np.reshape(run_test(experiment, problem, 'left', [1e3, 0], 0), (-1,2), order='C') #np.copy is removed
    #trialy_disp = np.reshape(run_test(experiment, problem, 'bottom', [0, 1e3], 0), (-1,2), order='C')
    #return combine_test_results([trialx_disp.flatten('F'), trialy_disp.flatten('F')]) #, trials1_disp

    #Sparse data (with sensors)
    _ = add_sensor(problem, 'left', test1_sensors_per_edge)
    trial1_disp = run_test(experiment, problem, 'left', [1e3, 0], 1).flatten()
    _ = add_sensor(problem, 'bottom', test2_sensors_per_edge)
    trial2_disp = run_test(experiment, problem, 'bottom', [0, 1e3], 1).flatten()
    return combine_test_results([trial1_disp, trial2_disp])

from numpy import linalg as LA
cost_function_values = []
total_model_error = []
displacement_model_error = []
#parameter_values = {'E_m':[], 'E_d':[]}
#parameter_values['E_m'].append(param[0])
#parameter_values['E_d'].append(param[1])
#sparsity_factor = 1e-6
def cost_function(param, sparsity_factor):
    # Function to calculate the cost function
    displacement_model = forward_model_run(param)  
    #print(sparsity_factor)
    #delta_displacement = displacement_model - displacement_data
    delta_displacement = (displacement_model - displacement_data)/(displacement_data + 1e-10)
    #print('Inferred Parameters',param)
    function_evaluation = np.dot(delta_displacement, delta_displacement) 
    cost_function_value = function_evaluation + sparsity_factor*LA.norm(param[np.array([1, 2, 3, 4, 5])], ord=1)
    displacement_model_error.append(function_evaluation)
    total_model_error.append(cost_function_value)
    return cost_function_value

#print(cost_function([0.41, 0.03, 0.26, 0.2, 0.1, 0.2]))

from scipy.optimize import minimize, least_squares, LinearConstraint

constraint_matrix = np.array([[1,-1, 0, 0, 0 ,0]]) # 0, 0 ,0
constraint = LinearConstraint(constraint_matrix, lb = [0])
start_point = np.array([0.9, 0.6, 0.32, 0.2, 0.4, 0.3 ])  #0.2, 0.4, 0.3
parameter_bounds = [(0, 1), (0, 1), (0, 0.45), (0, np.inf), (0, 1), (0, 1)] #   L-BFGS-B , 


#res = minimize(cost_function, start_point, method='trust-constr', bounds=parameter_bounds, constraints=[constraint],
#              options={'disp': True, 'gtol': 1e-16, 'xtol': 1e-10,},  ) #'ftol': 1e-10, 'xtol': 1e-16, 'barrier_tol': 1e-16,
#print(res.x) 
#print('Results', res.fun, res.grad, res.v, res.cg_stop_cond)
#print("Spring Stiffness:","{:e}".format(10**(12 - (12-6)*res.x[0])),  "{:e}".format(10**(12 - (12-6)*res.x[1])) )
#print("Inferred Values E_m, E_d, nu, G_12: \n",np.multiply(res.x, np.array([E_scaler, E_scaler,  1, G_12_scaler, 0, 0]))) #1, G_12_scaler
#print(p['G_12'])

""" import plotly.express as px
fig = px.line(x=[i for i in range(46,len(cost_function_values)+1)], y=cost_function_values[45:], markers=True, title='Cost Function Curve', log_y=True)
fig.update_layout(
    title_text='Cost Function Curve',
)
fig.show() """

import matplotlib.pyplot as plt
from matplotlib import cm


# Plotting the tendenacy of the parameters to tend to zero.
sp_factor = [1e-16, 1e-9, 1e-6, 1e-4, 1e-2, 0.1, 10, 1e2, 1e3]

inferred_parameters = np.zeros((len(sp_factor), len(start_point)))
for index, value in enumerate(sp_factor):
    print('#', index+1)
    res = minimize(cost_function, start_point, method='trust-constr', bounds=parameter_bounds, constraints=[constraint], args = (value),
                  options={'disp': True, 'gtol': 1e-16, 'xtol': 1e-10,}, )
    inferred_parameters[index] = res.x

import plotly.graph_objects as go
fig1 = go.Figure()
inferred_parameters_name = ['E_m', 'E_d', 'nu', 'G_12', 'K_x', 'K_y']

for i in range(inferred_parameters.shape[1]):
        fig1.add_trace(go.Scatter(x=sp_factor, y=[x for x in inferred_parameters[:,i]],
                        mode='markers',
                        name=inferred_parameters_name[i]))
fig1.add_hline(y=0.1, line_dash="dot")
fig1.update_xaxes(type="log")
fig1.update_yaxes(type="log")
fig1.update_layout(title="Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)",
    xaxis_title="Sparsity Factor",
    yaxis_title="Inferred Parameters (Log Scale)",
    legend_title="Parameters",)

fig1.update_traces(marker=dict(size=11,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig1.show()
fig1.write_html('Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)_RelativeError_final'+'.html')
np.savetxt('Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)_RelativeError_final".csv', inferred_parameters, delimiter=",")



#import plotly.graph_objects as go
#inferred_parameters = np.loadtxt('Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)_RelativeError', dtype=float, delimiter=",")
cf_value = []
for i in range(inferred_parameters.shape[0]):
    cf_value.append(cost_function(inferred_parameters[i], 0)) 

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sp_factor, y=cf_value,
                        mode='lines+markers',))
fig2.update_xaxes(type="log")
fig2.update_yaxes(type="log")
fig2.update_layout(title="Inferred Parameters Vs. Cost Function (No sparsity term, 1% Noise, Sparse Data)",
    xaxis_title="Sparsity Factor",
    yaxis_title="Cost Function (Log Scale)")
fig2.update_traces(marker=dict(size=11,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='lines+markers'))
fig2.show()
fig2.write_html('Inferred Parameters Vs. Cost Function (1% Noise, Sparse Data)_RelativeError_final'+'.html')
np.savetxt('Inferred Parameters Vs. Cost Function (1% Noise, Sparse Data)_RelativeError_final.csv', cf_value, delimiter=",")



""" # Plotting the model error + sparsity error
import plotly.graph_objects as go
iteration_no = np.arange(1, len(total_model_error)+1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=iteration_no, y=total_model_error,  
                    mode='lines+markers',
                    name='Total Model Error'))
fig.add_trace(go.Scatter(x=iteration_no, y=displacement_model_error,  
                    mode='lines+markers',
                    name='Displacement Model Error'))
#fig.update_layout(yaxis_type = "log")
fig.show() """

# N/m², m, kg, sec, N
#p['length'] = 5
#p['breadth'] = 1
#p['load'] = [1e6,0] #[0, -10] #
#p['rho'] = 7750
#p['g'] = 9.81
#p['E'] = 210e9 





































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


