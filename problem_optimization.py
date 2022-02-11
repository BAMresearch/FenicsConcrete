from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

from scipy.optimize import differential_evolution, minimize

set_log_level(30)

def tempeature_optimization_1(x, *args):
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

    duration = time # bad result!
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
            print(f'Max temperature exceeded! Stopping computation at t = {t/60/60}h.')
            T_limit_reached = True
            break
        # check for DOH limit
        if problem.sensors[0].data[-1][1] > DOH_min:
            print(f'Min DOH requirement reached! Stopping computation at t = {t/60/60}h.')
            # first time!!! -> problem with interpolation!!!
            if not DOH_min_reached:
                data2 = problem.sensors[0].data[-1]
                # value of step before last
                data1 = problem.sensors[0].data[-2]
            DOH_min_reached = True

        t += dt

    # output as hours
    if not T_limit_reached or not DOH_min_reached:
        print(f'DOH min has not been reached in {t/60/60}h.')
    else:
        print('T_limit_reached:',T_limit_reached)
        print('DOH_min_reached:',DOH_min_reached)
        # get duration based on interpolation, assuming linear DOH evolution at each timestep
        # value of last step
        duration = (data2[0]-data1[0])/(data2[1]-data1[1])*((DOH_min-data1[1])) + data1[0]

    return duration/60.0/60.0


def tempeature_optimization(x, *args):

    n_vars = args[0]
    T_limit = args[1]
    DOH_min = args[2]
    dt_in_min = args[3] # timestep in minues
    time_in_h = args[4] # max time in hours

    if n_vars == 1:
        bc_temp = x[0]  # boundary temp
        T_0 = args[5]  # initial concrete temperature
    elif n_vars == 2:
        bc_temp = x[0]  # boundary temp
        T_0 = x[1]  # initial concrete temperature
    else:
        raise Exception('wrong n_vars input (args[0]), only 1 or 2')

    # parameters = None # using the current default values
    parameters = concrete_experiment.Parameters()
    parameters['T_0'] = T_0  # inital concrete temperature
    parameters['T_bc1'] = bc_temp  # temperature boundary value 1

    experiment = concrete_experiment.get_experiment('ConcreteCube', parameters)
    # TODO fix output!!!
    filename = f'output_T_bc_{bc_temp}_T_0_{T_0}'
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

    duration = time*bc_temp/10# bad result! TODO better option!
    #duration = time*2# bad result! TODO better option!
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
            T_limit_reached = True
            break
        # check for DOH limit
        # need to compute further because temperature could be triggered after doh reached??
        if problem.sensors[0].data[-1][1] > DOH_min:
            # TODO: could make a check if we passed max temp
            print(f'Min DOH requirement reached!')
            # first time!!!
            if not DOH_min_reached:
                duration = t # current time
                data2 = problem.sensors[0].data[-1]
                # value of step before last
                data1 = problem.sensors[0].data[-2]


            DOH_min_reached = True


            #break

        t += dt

    # output as hours
    if T_limit_reached:
        print(f'Max temperature exceeded! Stopping computation at t = {t/60/60}h.')
        # higher temp, worse numbers!!!
        duration = time * bc_temp / 10  # bad result! TODO better option!
    elif not DOH_min_reached:
        # lower temperature, worse numbers!!!
        duration = time * (T_limit-bc_temp)/10# bad result! TODO better option!
        print(f'DOH min has not been reached in {t/60/60}h.')
    else:
        # get duration based on interpolation, assuming linear DOH evolution at each timestep
        duration = (data2[0]-data1[0])/(data2[1]-data1[1])*((DOH_min-data1[1])) + data1[0]

    return duration/60.0/60.0


#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------

# settings!
T_bc = 40
T_0 = 20.2
T_limit = 70
T_min = 10
DOH_min = 0.7
dt_in_min = 60
time_in_h = 48

n_vars = 2
# find optimum

variables = [T_bc,T_0]
args = (n_vars,T_limit,DOH_min,dt_in_min,time_in_h)

bounds = [(T_min,T_limit),(T_min,T_limit)]
x0 = [20,20]
result = minimize(tempeature_optimization,x0, bounds = bounds, args= args, method='Nelder-Mead', tol = 0.1)

# set optimum
T_bc = result.x[0]
T_0 = result.x[1]

variables = [T_bc,T_0]
duration = tempeature_optimization(variables, n_vars, T_limit, DOH_min, dt_in_min, time_in_h)
print(f'Result for T_boundary = {T_bc}, T_0 = {T_0}: Duration = {duration} h')
