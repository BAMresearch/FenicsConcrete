
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

    problem.pv_plot()

    # last measurement
    
    #return sensor_output

    return problem.sensors
import math
p = fenicsX_concrete.Parameters()  # using the current default values
p['problem'] = 'cantilever_beam' #'cantilever_beam' #

# N/m², m, kg, sec, N
p['rho'] = 7750
p['g'] = 9.81
p['E'] = 210e9
p['length'] = 1
p['breadth'] = 0.2
#p['load'] = 1000#-10e8
#p['k_x'] = 1e15
#p['k_y'] = 1e13
p['k_spring'] = 1e11
# MPa, mm, kg, sec, N
#p['rho'] = 7750e-9 #kg/mm³
#p['g'] = 9.81#e3 #mm/s² for units to be consistent g must be given in m/s².
#p['E'] = 210e3 #N/mm² or MPa
#p['length'] = 1000
#p['breadth'] = 200
#p['load'] = 100e-6 #N/mm²

p['nu'] = 0.28
p['num_elements_length'] = 30
p['num_elements_breadth'] = 20
p['dim'] = 2

#Defining sensor positions
sensor = []
sensor_pos_x = []
number_of_sensors = 20
for i in range(number_of_sensors):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/20*(i+1), 0.5*p['breadth'], 0]])))
    sensor_pos_x.append(p['length']/20*(i+1))

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

""" import matplotlib.pyplot as plt
#import numpy as np
plt.plot(sensor_pos_x, displacement_data[:,1])
plt.show() """