from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

#import probeye
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.scipy_.solver import ScipySolver


# initiate material problem
material_problem = concrete_problem.ConcreteThermoMechanical()

# get function over time!


# get hydration over time!!!
# this method *must* be provided by the user
dt = 60*30
T = 20
time_list = np.arange(0,60*60*48,dt)
parameter = {}
parameter['B1'] = 0.000361
parameter['B2'] = 0.00023
parameter['eta'] = 4.81
parameter['alpha_max'] = 0.875
parameter['E_act'] = 32995
parameter['T_ref'] = 25
parameter['Q_pot'] = 450000

# get the respective function
hydration_fkt = material_problem.get_heat_of_hydration_ftk()

heat_list, doh = hydration_fkt(T, time_list, dt, parameter)

##print(doh)

# get youngs modulus!!!
# get the respective function
E_fkt = material_problem.get_E_alpha_fkt()

parameters = {}
parameters['alpha_t'] = 0.09
parameters['E_inf'] = 54.2
parameters['alpha_0'] = 0.06
parameters['a_E'] = 0.4

E_list = E_fkt(doh,parameters)

fc_fkt = material_problem.get_X_alpha_fkt()

parameters['X_inf'] = 62.1
parameters['a_X'] = 1.2

fc_list = fc_fkt(doh, parameters)

ft_fkt = material_problem.get_X_alpha_fkt()

parameters['X_inf'] = 4.67
parameters['a_X'] = 1.0

ft_list = ft_fkt(doh, parameters)

plt.plot(time_list,doh)
plt.show()

plt.plot(time_list,E_list)
plt.show()

plt.plot(time_list,fc_list)
plt.show()

plt.plot(time_list,ft_list)
plt.show()
# plt.plot(alpha_list,fc_list)
# plt.show()