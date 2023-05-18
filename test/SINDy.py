
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
p['k_x'] = 1e15
p['k_y'] = 1e13
p['K_torsion'] = 1e11
p['degree'] = 2             # polynomial degree


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

# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs.
# 3: Torsion Springs
p['uncertainties'] = [0]

#Defining sensor positions
sensor = []
sensor_pos_x = []
number_of_sensors = 25
for i in range(number_of_sensors):
    sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[p['length']/number_of_sensors*(i+1), p['breadth'], 0]])))
    #sensor.append(fenicsX_concrete.sensors.StrainSensor(np.array([[p['length']/20*(i+1), 0.5*p['breadth'], 0]])))
    sensor.append(fenicsX_concrete.sensors.StrainSensor(np.array([[p['length']/number_of_sensors*(i+1), p['breadth'], 0]])))
    sensor.append(fenicsX_concrete.sensors.DisplacementDoubleDerivativeSensor(np.array([[p['length']/number_of_sensors*(i+1), p['breadth'], 0]])))
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
displacement_double_derivative_model = np.zeros((number_of_sensors,8))

displacement_data = collect_sensor_solutions(solution, disp_model, fenicsX_concrete.sensors.DisplacementSensor)
strain_data = collect_sensor_solutions(solution, strain_model, fenicsX_concrete.sensors.StrainSensor)
displacement_double_derivative_data = collect_sensor_solutions(solution, displacement_double_derivative_model, fenicsX_concrete.sensors.DisplacementDoubleDerivativeSensor)

""" import plotly.express as px
fig = px.line(x=sensor_pos_x, y=strain_data[:,3], markers=True, title='Strain Curve')
fig.show()
"""

#import plotly.express as px
#fig = px.line(x=sensor_pos_x, y=displacement_double_derivative_data[:,0], markers=True, title='Displacement Double Derivative Curve')
#fig.show()

""" import matplotlib.pyplot as plt
#import numpy as np
plt.plot(sensor_pos_x, displacement_data[:,1])
plt.show() """

""" F_matrix = np.zeros((number_of_sensors, 5))
F_matrix[:,0] = -displacement_double_derivative_data[:,5] 
F_matrix[:,1] = -displacement_double_derivative_data[:,3]
F_matrix[:,2] = -displacement_double_derivative_data[:,1]
F_matrix[:,3] = -displacement_double_derivative_data[:,4]
F_matrix[:,4] = 1#-p['rho']*p['g']
LHS_vector = np.zeros((number_of_sensors,2))
LHS_vector[:,0] = displacement_double_derivative_data[:,0]
LHS_vector[:,1] = displacement_double_derivative_data[:,7] #+ p['rho']*p['g'] """


F_matrix = np.zeros((number_of_sensors, 2))
F_matrix[:,0] = -displacement_double_derivative_data[:,5] 
F_matrix[:,1] = -displacement_double_derivative_data[:,3]
#F_matrix[:,4] = 1#-p['rho']*p['g']
LHS_vector = np.zeros((number_of_sensors,2))
LHS_vector[:,0] = displacement_double_derivative_data[:,0]
#
#
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(F_matrix, LHS_vector)
print(clf.coef_)
#
clf2 = linear_model.Ridge(alpha=0.0)
clf2.fit(F_matrix, LHS_vector)
print(clf2.coef_)

""" F_matrix = np.zeros((number_of_sensors, 5))
F_matrix[:,0] = -displacement_double_derivative_data[:,5] 
F_matrix[:,1] = -displacement_double_derivative_data[:,3]
F_matrix[:,2] = -displacement_double_derivative_data[:,1]
F_matrix[:,3] = -displacement_double_derivative_data[:,4]
F_matrix[:,4] = -p['rho']*p['g']
LHS_vector = np.zeros((number_of_sensors,2))
LHS_vector[:,0] = displacement_double_derivative_data[:,0]
LHS_vector[:,1] = displacement_double_derivative_data[:,7] 

import pysindy as ps
control_variables = np.copy(F_matrix)
optimiser = ps.STLSQ(threshold=0.07, fit_intercept=False)
feature_library_1=feature_library=ps.PolynomialLibrary(degree=2) #ps.FourierLibrary(n_frequencies=3)
feature_library_2=feature_library=ps.FourierLibrary(n_frequencies=1) 
model = ps.SINDy(feature_names=['d','dxzUz', 'dzzUx', 'dzxUx', 'dzzUz', 'b'], feature_library = feature_library_1 + feature_library_2) #feature_library=feature_lib

print(model)
model.fit(f, u=control_variables,t=x)
print(model.get_feature_names())
model.print()  """