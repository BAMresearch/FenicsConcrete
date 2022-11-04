
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize

def simple_setup(p, sensor):
    parameters = fenicsX_concrete.Parameters()  # using the current default values

    #parameters['log_level'] = 'WARNING'
    parameters['bc_setting'] = 'free'
    parameters['mesh_density'] = 10

    parameters = parameters + p

    experiment = fenicsX_concrete.concreteSlabExperiment(parameters)         # Specifies the domain, discretises it and apply Dirichlet BCs

    problem = fenicsX_concrete.LinearElasticity(experiment, parameters)      # Specifies the material law and weak forms.
    #print(help(fenics_concrete.LinearElasticity))

    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])

    #problem.experiment.apply_load_bc(displacement)

    problem.solve()  # solving this

    #print(problem.displacement([0,2]))

    #problem.pv_plot()

    # last measurement
    
    #return sensor_output

    return problem.sensors

p = fenicsX_concrete.Parameters()  # using the current default values
p['E'] = 100
p['nu'] = 0.2
p['length'] = 1
p['breadth'] = 0.2
p['num_elements_length'] = 20
p['num_elements_breadth'] = 10
p['dim'] = 2
#displacement = -3

sensor = []
number_of_sensors = 20
for i in range(number_of_sensors):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.1, 0]])))
    #sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1, 0.02*(i+1),  0]])))

# Synthetic data generation
solution = simple_setup(p, sensor)
number_of_sensors =20

def collect_sensor_solutions(model_solution, total_sensors):
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model
    #print(measured[i].data[-1])

displacement_data = collect_sensor_solutions(solution, number_of_sensors)
max_disp_value = np.amax(np.absolute(displacement_data[:,1]))
#print(displacement_data)

sigma_error = 0.1*max_disp_value
sigma_prior = 0.1*max_disp_value

np.random.seed(42) 
distortion = np.random.normal(0, sigma_error, (number_of_sensors,2))
#print(distortion) 

displacement_measured = displacement_data + distortion
#print(displacement_measured)

#Cost function plot
def cost_func_and_jac(optimised_parameters, measured_data, regularisation_constant = 0, placed_sensors = sensor):
    #Cost function evaluation
    p = fenicsX_concrete.Parameters()
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1]
    p['length'] = 1
    p['breadth'] = 0.2
    p['num_elements_length'] = 20
    p['num_elements_breadth'] = 10
    p['dim'] = 2
    sim_output = simple_setup(p, placed_sensors)
    predicted_displacement = collect_sensor_solutions(sim_output, 20)
    del sim_output
    delta = predicted_displacement - measured_data

    parameters_vector = np.array([p.E, p.nu])
    

    normed_vector = parameters_vector#parameters_vector/np.linalg.norm(parameters_vector)

    cost_function_value = np.dot(delta[:,1], delta[:,1]) + regularisation_constant*np.dot(normed_vector,normed_vector)
    print(parameters_vector, cost_function_value)
    return cost_function_value


# Parameter vs cost-function plot
E_buildup = np.linspace(80,120,30)
E_samples = np.zeros((270,))
for index in range (len(E_buildup)):
    E_samples[9*index:9*(index+1)] = E_buildup[index]

nu_buildup = np.linspace(0.1,0.45,9)
nu_samples = np.zeros((270,))
for index in range (30):
    nu_samples[9*index:9*(index+1)] = nu_buildup

parameters = np.zeros((270,2))
parameters[:,0] = E_samples
parameters[:,1] = nu_samples

cost_func_val = np.zeros((270,))

for i in range(270):
    print('sim'+str(i+1))
    cost_func_val[i] = cost_func_and_jac(parameters[i], displacement_measured) 
    #print(cost_func_val)

import plotly.express as plx
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Surface(z=cost_func_val.reshape(30,9), x=E_buildup, y=nu_buildup))
fig.update_layout(
    title="Parameters Vs. Cost-Function",
    xaxis_title="Youngs Modulus",
    yaxis_title="Poissons Ratio")
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))

fig.show()


#Inverse Problem
""" def cost_func_and_jac(optimised_parameters, measured_data, regularisation_constant = 0, placed_sensors = sensor):
    #Cost function evaluation
    p = fenicsX_concrete.Parameters()
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1]
    p['length'] = 1
    p['breadth'] = 0.2
    p['num_elements_length'] = 20
    p['num_elements_breadth'] = 10
    p['dim'] = 2
    predicted_displacement = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    delta = predicted_displacement - measured_data

    parameters_vector = np.array([p.E, p.nu])
    print(parameters_vector, np.dot(delta[:,1], delta[:,1]))

    cost_function_value = np.dot(delta[:,1], delta[:,1]) + regularisation_constant*np.dot(parameters_vector,parameters_vector)

    
    #Jacobian evaluation
    delta_E = 0.001
    p['E'] = optimised_parameters[0] + delta_E
    measured_Eplus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    p['E'] = optimised_parameters[0] - delta_E
    measured_Eminus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    derv_E = (measured_Eplus[:,1] - measured_Eminus[:,1])/(2*delta_E)

    delta_nu = 0.0001
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1] + delta_nu
    measured_nuplus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    p['nu'] = optimised_parameters[1] - delta_nu
    measured_numinus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    derv_nu = (measured_nuplus[:,1]  - measured_numinus[:,1])/(2*delta_nu)
    print('jacobian called')
    jacobian_value1= 2*np.dot(delta[:,1],derv_E)
    jacobian_value2= 2*np.dot(delta[:,1],derv_nu)
    jacobian_value = np.array([jacobian_value1, jacobian_value2])
    return cost_function_value, jacobian_value

def hessian_function(optimised_parameters, measured_data=displacement_measured, placed_sensors = sensor):
    #Cost function evaluation
    p = fenicsX_concrete.Parameters()
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1]
    p['length'] = 1
    p['breadth'] = 0.2
    p['num_elements_length'] = 20
    p['num_elements_breadth'] = 10
    p['dim'] = 2
    predicted_displacement = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    delta_E = 0.001
    p['E'] = optimised_parameters[0] + delta_E
    measured_Eplus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    p['E'] = optimised_parameters[0] - delta_E
    measured_Eminus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    derv_E = (measured_Eplus[:,1]  - measured_Eminus[:,1])/(2*delta_E)
    double_derv_E = (measured_Eplus[:,1]  - 2*predicted_displacement[:,1]  + measured_Eminus[:,1] )/delta_E**2

    delta_nu = 0.0001
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1] + delta_nu
    measured_nuplus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    p['nu'] = optimised_parameters[1] - delta_nu
    measured_numinus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    derv_nu = (measured_nuplus[:,1]  - measured_numinus[:,1])/(2*delta_nu)
    double_derv_nu = (measured_nuplus[:,1]  - 2*predicted_displacement[:,1]  + measured_numinus[:,1] )/delta_nu**2

    p['E'] = optimised_parameters[0] + delta_E
    p['nu'] = optimised_parameters[1] + delta_nu
    measured_Eplus_nuplus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    p['E'] = optimised_parameters[0] - delta_E
    p['nu'] = optimised_parameters[1] - delta_nu
    measured_Eminus_numinus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    p['E'] = optimised_parameters[0] + delta_E
    p['nu'] = optimised_parameters[1] - delta_nu
    measured_Eplus_numinus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    p['E'] = optimised_parameters[0] - delta_E
    p['nu'] = optimised_parameters[1] + delta_nu
    measured_Eminus_nuplus = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)

    mixed_derv_E_nu = (measured_Eminus_numinus[:,1]  + measured_Eplus_nuplus[:,1]  - measured_Eplus_numinus[:,1]  - measured_Eminus_nuplus[:,1] )/(4*delta_E*delta_nu)

    hessian = np.zeros((2,2))
    delta = predicted_displacement - measured_data
    #print('hessian called')
    hessian[0][0] = 2*(np.dot(derv_E,derv_E) + np.dot(delta[:,1],double_derv_E))
    hessian[1][1] = 2*(np.dot(derv_nu,derv_nu) + np.dot(delta[:,1],double_derv_nu))
    hessian[0][1] = 2*(np.dot(derv_E,derv_nu) + np.dot(delta[:,1],mixed_derv_E_nu))
    hessian[1][0] = 2*(np.dot(derv_E,derv_nu) + np.dot(delta[:,1],mixed_derv_E_nu))
    return hessian
 """

#Deterministic
""" def cost_func(optimised_parameters, measured_data, regularisation_constant = 0, placed_sensors = sensor):
    #Cost function evaluation
    p = fenicsX_concrete.Parameters()
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1]
    p['length'] = 1
    p['breadth'] = 0.2
    p['num_elements_length'] = 20
    p['num_elements_breadth'] = 10
    p['dim'] = 2
    predicted_displacement = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    delta = predicted_displacement - measured_data

    parameters_vector = np.array([p.E, p.nu])
    print(parameters_vector, np.dot(delta[:,1], delta[:,1]))

    cost_function_value = np.dot(delta[:,1], delta[:,1]) + regularisation_constant*np.dot(parameters_vector,parameters_vector)

    return cost_function_value """

#Probabilistic

""" def cost_func(optimised_parameters, measured_data, std_dev_error = sigma_error, std_dev_prior = 20 ,placed_sensors = sensor):
    #Cost function evaluation
    p = fenicsX_concrete.Parameters()
    p['E'] = optimised_parameters[0]
    p['nu'] = optimised_parameters[1]
    p['length'] = 1
    p['breadth'] = 0.2
    p['num_elements_length'] = 20
    p['num_elements_breadth'] = 10
    p['dim'] = 2
    predicted_displacement = collect_sensor_solutions(simple_setup(p, placed_sensors), 20)
    delta = predicted_displacement - measured_data

    parameters_vector = np.array([p.E, p.nu])
    print(parameters_vector, np.dot(delta[:,1], delta[:,1]))


    cost_function_value = np.dot(delta[:,1], delta[:,1]) + (std_dev_error**2/std_dev_prior**2)*np.dot(parameters_vector,parameters_vector)

    return cost_function_value


print('Cost function optimisation started...')
result = optimize.minimize(cost_func, (80, 0.2), args=(displacement_measured),  method='powell') #Newton-CG
#result = optimize.minimize(cost_func_and_jac, (80, 0.4), args=(displacement_measured), jac=True, hess=hessian_function, method='Newton-CG') #Newton-CG
print(result) """
