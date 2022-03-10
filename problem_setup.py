from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem
import time as timer
#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------
start = timer.time()

parameters = concrete_experiment.Parameters() # using the current default values
# boundary values...
parameters['dim'] = 2  # inital concrete temperature
parameters['mesh_density'] = 10  # inital concrete temperature
#parameters['mesh_setting'] = 'left'  # inital concrete temperature
parameters['mesh_setting'] = 'left/right'  # inital concrete temperature
parameters['pol_degree'] = 2 # inital concrete temperature

experiment = concrete_experiment.get_experiment('ConcreteCube',parameters)
problem = concrete_problem.ConcreteThermoMechanical(experiment,parameters)

dohhom_sensor = concrete_experiment.DOHHomogeneitySensor()
problem.add_sensor(dohhom_sensor)





# data for time stepping
#time steps
dt = 60*20 # time step
hours = 10
time = hours*60*60         # total simulation time in s
# set time step
problem.set_timestep(dt) # for time integration scheme

#initialize time
t = dt # first time step time

# only one timestep for testing, moin
while t <= time: # time

    print('time =', t)
    # solve temp-hydration-mechanics
    problem.solve(t=t) # solving this

    # plot fields
    problem.pv_plot(t=t)

    #print(problem(u_sensor))
    # prepare next timestep
    t += dt
#
# print(problem.sensors[0].data)
# print(problem.sensors[1].data)
# print(problem.sensors[2].data)
end = timer.time()
print('duration:',end-start)
