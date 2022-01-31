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
# TODO sensors!!!

# data for time stepping
#time steps
dt = 60*20 # time step
hours = 50
time = hours*60*60         # total simulation time in s
# set time step
problem.set_timestep(dt) # for time integration scheme

#initialize time
t = dt # first time step time

while t <= time:

    print('time =', t)
    # solve temp-hydration-mechanics
    problem.solve() # solving this

    # plot fields
    problem.pv_plot(t=t)
    # prepare next timestep
    t += dt


