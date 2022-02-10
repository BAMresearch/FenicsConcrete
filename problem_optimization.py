from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem




def tempeature_optimization(x, *args):
    bc_temp = x[0] # boundary temp
    T_0 = args[4]  # initial concrete temperature
    T_limit = args[0]
    DOH_min = args[1]
    dt_in_min = args[2] # timestep in minues
    time_in_h = args[3] # max time in hours

    # parameters = None # using the current default values
    parameters = concrete_experiment.Parameters()
    parameters['T_0'] = T_0  # inital concrete temperature
    parameters['T_bc1'] = bc_temp  # temperature boundary value 1

    experiment = concrete_experiment.get_experiment('ConcreteCube', parameters)
    # TODO fix output!!!
    filename = f'output_T_bc_{bc_temp}'
    problem = concrete_problem.ConcreteThermoMechanical(experiment, parameters, pv_name=filename)

    min_DOH_sensor = concrete_experiment.MinDOHSensor()
    max_T_sensor = concrete_experiment.MaxTemperatureSensor()

    problem.add_sensor(min_DOH_sensor)
    problem.add_sensor(max_T_sensor)

    # data for time stepping
    # time steps
    dt = 60 * dt_in_min  # time step
    hours = time_in_h
    time = hours * 60 * 60  # total simulation time in s
    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    # initialize time
    t = dt  # first time step time

    T_limit_reached = False
    DOH_min_reached = False

    duration = time*10 # bad result!
    while t <= time:

        print('time =', t)
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this

        # plot fields
        problem.pv_plot(t=t)

        # print(problem(u_sensor))
        # prepare next timestep

        # check last max temp value
        if problem.sensors[1].data[-1][1] > T_limit:
            print(f'Max temperature exceeded! Stopping computation at t={t - dt}.')
            T_limit_reached = True
            break
        # check for DOH limit
        if problem.sensors[0].data[-1][1] > DOH_min:
            print(f'Min DOH requirement reached! Stopping computation at t={t - dt}.')
            DOH_min_reached = True
            duration = t # current time
            break

        t += dt

    return duration

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------






#while
data = []
T_limit = 70
DOH_min = 0.8
#trying to optimize the boundary!
# very simplified "optimization"
# max temp = 70
# max difference = 19

# highest constant boundary temp, without exceeding 70°C
T_0 = 30
bc_temp = 20.0 # initial temperature
n = 1 # exponent
bc_data = []
#for bc_temp in [20,30,40,50,60,70]:
while n < 3 :
    t_increae = 10.0**(2-n)


#parameters = None # using the current default values
    parameters = concrete_experiment.Parameters()
    parameters['T_0'] = T_0  # inital concrete temperature
    parameters['T_bc1'] = bc_temp  # temperature boundary value 1

    experiment = concrete_experiment.get_experiment('ConcreteCube',parameters)
    filename = f'output_T_bc_{bc_temp}'
    problem = concrete_problem.ConcreteThermoMechanical(experiment,parameters,pv_name = filename)

    min_DOH_sensor = concrete_experiment.MinDOHSensor()
    max_T_sensor = concrete_experiment.MaxTemperatureSensor()

    problem.add_sensor(min_DOH_sensor)
    problem.add_sensor(max_T_sensor)


    # data for time stepping
    #time steps
    dt = 60*60 # time step
    hours = 48
    time = hours*60*60         # total simulation time in s
    # set time step
    problem.set_timestep(dt) # for time integration scheme


    #initialize time
    t = dt # first time step time

    T_limit_reached = False
    DOH_min_reached = False
    while t <= time:

        print('time =', t)
        # solve temp-hydration-mechanics
        problem.solve(t=t) # solving this

        # plot fields
        problem.pv_plot(t=t)

        #print(problem(u_sensor))
        # prepare next timestep
        t += dt

        # check last max temp value
        if problem.sensors[1].data[-1][1] > T_limit:
            print(f'Max temperature exceeded! Stopping computation at t={t-dt}.')
            T_limit_reached = True
            break
        # check for DOH limit
        if problem.sensors[0].data[-1][1] > DOH_min:
            print(f'Min DOH requirement reached! Stopping computation at t={t-dt}.')
            DOH_min_reached = True
            break

    # max temperature
    problem_data = np.asmatrix(problem.sensors[1].data)
    T_data = problem_data[:, 1]
    T_max = np.amax(T_data)
    # min DOH
    DOH_min = problem.sensors[0].data[-1][1]
    duration = problem.sensors[0].data[-1][0]

    if T_limit_reached:
        n += 1
        bc_temp -= t_increae - t_increae/10
    else:
        data.append([bc_temp,T_max,DOH_min,duration])
        bc_temp += t_increae




for set in data:
    print(set)