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

    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])

    #problem.experiment.apply_load_bc(displacement)

    problem.solve()  # solving this

    problem.pv_plot()

    # last measurement
    
    #return sensor_output

    return problem.sensors

import math
p = fenicsX_concrete.Parameters()  # using the current default values
p['problem'] = 'cantilever_beam' #'cantilever_beam' #

# N/m², m, kg, sec, N
p['rho'] = 7750
p['nu'] = 0.28
p['g'] = 9.81
p['E'] = 210e9
p['length'] = 1
p['breadth'] = 0.2
#p['load'] = 1000#-10e8
p['k_x'] = 1e15
p['k_y'] = 1e13
p['K_torsion'] = 1e11
p['degree'] = 2 

# MPa, mm, kg, sec, N
#p['rho'] = 7750e-9 #kg/mm³
#p['g'] = 9.81#e3 #mm/s² for units to be consistent g must be given in m/s².
#p['E'] = 210e3 #N/mm² or MPa
#p['length'] = 1000
#p['breadth'] = 200
#p['load'] = 100e-6 #N/mm²

p['num_elements_length'] = 20
p['num_elements_breadth'] = 20
p['dim'] = 2

# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs.
# 3: Torsion Springs
p['uncertainties'] = [0]

#Defining sensor positions
sensor = []
sensor_pos_x = []
number_of_sensors = 20
for i in range(number_of_sensors):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/number_of_sensors*(i+1), 0.5*p['breadth'], 0]])))
    sensor.append(fenicsX_concrete.sensors.StrainSensor(np.array([[p['length']/number_of_sensors*(i+1), 0.5*p['breadth'], 0]])))
    sensor_pos_x.append(p['length']/number_of_sensors*(i+1))

# Synthetic data generation
solution = simple_setup(p, sensor)

def collect_sensor_solutions(model_solution, field_data, field_type):
    counter=0
    for i in model_solution:
        if isinstance(model_solution[i], field_type):
            field_data[counter] = model_solution[i].data[-1]
            counter += 1
    return field_data
    #print(measured[i].data[-1])

disp_model = np.zeros((number_of_sensors,2))
strain_model = np.zeros((number_of_sensors,4))

displacement_data = collect_sensor_solutions(solution, disp_model, fenicsX_concrete.sensors.DisplacementSensor)
strain_data = collect_sensor_solutions(solution, strain_model, fenicsX_concrete.sensors.StrainSensor)

import plotly.express as px
fig = px.line(x=sensor_pos_x, y=strain_data[:,0], markers=True, title='Strain Curve')
fig.update_layout(
    title_text='Vertical Displacement Curve'
)
fig.show()

""" import matplotlib.pyplot as plt
#import numpy as np
plt.plot(sensor_pos_x, displacement_data[:,1])
plt.show() """
