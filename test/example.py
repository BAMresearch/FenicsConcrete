
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import matplotlib.pyplot as matplot

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
sensor_pos_x = []
number_of_sensors = 20
for i in range(number_of_sensors):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[1/20*(i+1), 0.1, 0]])))
    sensor_pos_x.append(1/20*(i+1))
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

import plotly.express as px
fig = px.line(x=sensor_pos_x, y=displacement_data[:,1])
fig.update_layout(
    title_text='Vertical Displacement Curve'
)
fig.show()