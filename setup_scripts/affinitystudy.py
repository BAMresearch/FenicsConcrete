from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------

parameters = None # using the current default values
experiment = concrete_experiment.get_experiment('ConcreteCube',parameters)
problem = concrete_problem.ConcreteThermoMechanical(experiment,parameters)
#
plt.figure()
plt.ticklabel_format(style='scientific',scilimits=(0,0))
plt.ylabel('Affinity')
plt.xlabel('Degree of Hydration')

# vary B2
data = []
p_list = [0.0,0.01,0.03]
for parameter in p_list:
    problem.temperature_problem.mat.alpha_max = 1
    problem.temperature_problem.mat.B2 = parameter
    alpha, a = problem.temperature_problem.get_affinity()
    data.append(a)
problem.temperature_problem.mat.B2 = 0.0024229
#
for i in range(len(data)):
   plt.plot(alpha,data[i],color='#DCE319FF',linestyle='dashed') # gelb

# eta
p_list = [3.8,4.7,8]
data = []
for parameter in p_list:
    problem.temperature_problem.mat.alpha_max = 1
    problem.temperature_problem.mat.eta = parameter
    alpha, a = problem.temperature_problem.get_affinity()
    data.append(a)
problem.temperature_problem.mat.eta = 5.554

for i in range(len(data)):
     plt.plot(alpha,data[i],color='#39568CFF',linestyle='dashed')
#B1
p_list = [1.6e-4,2.5e-4,3.8e-4]
data = []
for parameter in p_list:
    problem.temperature_problem.mat.alpha_max = 1
    problem.temperature_problem.mat.B1 = parameter
    alpha, a = problem.temperature_problem.get_affinity()
    data.append(a)
problem.temperature_problem.mat.B1 = 2.916e-4
#
for i in range(len(data)):
    plt.plot(alpha,data[i],color='#73D055FF',linestyle='dashed')

# optimum
problem.temperature_problem.mat.alpha_max = 1
alpha, a = problem.temperature_problem.get_affinity()
plt.plot(alpha, a, color='black', linewidth=3)




plt.tight_layout()
plt.savefig('affinity_variation.png')
plt.show()