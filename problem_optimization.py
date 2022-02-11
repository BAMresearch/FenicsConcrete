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
            DOH_min_reached = True
            duration = t # current time


            break

        t += dt

    # output as hours
    if not T_limit_reached and not DOH_min_reached:
        print(f'DOH min has not been reached in {t/60/60}h.')
    else:
        # get duration based on interpolation, assuming linear DOH evolution at each timestep
        print(problem.sensors[0].data)
        # value of last step
        data2 = problem.sensors[0].data[-1]
        # value of step before last
        data1 = problem.sensors[0].data[-2]
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

setting = 0

if setting == 0:
    # manuel set optimum
    dt_in_min = 60
    T_bc = 49
    T_0 = 20.2
    n_vars = 1

    variables_2 = [T_bc,T_0]
    duration = tempeature_optimization(variables_2, n_vars, T_limit, DOH_min, dt_in_min, time_in_h, T_0)
    print(f'Result for T_boundary = {T_bc}, T_0 = {T_0}: Duration = {duration} h')

if setting == 1:
    # manuel set optimum
    dt_in_min = 60
    T_bc = 49
    T_0 = 20.2

    variables_1 = [T_bc]
    duration = tempeature_optimization_1(variables_1, T_limit, DOH_min, dt_in_min, time_in_h, T_bc)
    print(f'Result for T_boundary = {T_bc}, T_0 = {T_0}: Duration = {duration} h')




if setting == 2:


    variables_1 = [T_bc]
    args_1 = (T_limit,DOH_min,dt_in_min,time_in_h,T_0)

    bounds_1 = [(T_min,T_limit)]
    x0 = [42]
    result = minimize(tempeature_optimization_1,x0, bounds = bounds_1, args= args_1, method='Nelder-Mead', options={'xatol': 0.1})
    #result = minimize(tempeature_optimization_1, x0 = x0, bounds = bounds_1, args= args_1)

    variables_1 = [result.x[0]]
    print('Duration 1: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',result.x[0])
    T_bc = 49
    variables_1 = [T_bc]
    print('Duration 2: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',T_bc)

    #variables_1 = [result.x]
    #print('Duration: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',result.x)


#
# elif setting == 1:
#     variables_1 = [T_bc]
#     args_1 = (T_limit,DOH_min,dt_in_min,time_in_h,T_0)
#
#     bounds_1 = [(T_min,T_limit)]
#     x0 = [30]
#     result = differential_evolution(tempeature_optimization_1, bounds = bounds_1, args= args_1, workers= -1)
#     #result = minimize(tempeature_optimization_1, x0 = x0, bounds = bounds_1, args= args_1)
#
#     variables_1 = [result.x[0]]
#     print('Duration: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',result.x[0])
#     T_bc = 49
#     variables_1 = [T_bc]
#     print('Duration: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',T_bc)
#
# elif setting == 10:
#     variables_1 = [T_bc]
#     args_1 = (T_limit,DOH_min,dt_in_min,time_in_h,T_0)
#
#     bounds_1 = [(T_min,T_limit)]
#     x0 = [42]
#     result = minimize(tempeature_optimization_1,x0, bounds = bounds_1, args= args_1, method='Nelder-Mead', options={'xatol': 0.1})
#     #result = minimize(tempeature_optimization_1, x0 = x0, bounds = bounds_1, args= args_1)
#
#     variables_1 = [result.x[0]]
#     print('Duration 1: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',result.x[0])
#     T_bc = 49
#     variables_1 = [T_bc]
#     print('Duration 2: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',T_bc)
#
#     #variables_1 = [result.x]
    #print('Duration: ',tempeature_optimization_1(variables_1,T_limit,DOH_min,dt_in_min,time_in_h,T_0),'T_bc: ',result.x)

# # now setup multivariable problem
# elif setting == 2:
#     # very narrow bounds
#     variables_2 = [T_bc,T_0]
#     args_2 = (T_limit,DOH_min,dt_in_min,time_in_h)
#     bounds_2 = [(30,35),(20,25)]
#     result = differential_evolution(tempeature_optimization_2, bounds = bounds_2, args= args_2, workers= -1)
#     print('result:',result.x)
#     variables_2 = [result.x]
#     print('Duration: ',tempeature_optimization_2(variables_2,T_limit,DOH_min,dt_in_min,time_in_h),'T_bc: ',result.x)


#
#
# elif setting == 3:
#     # brute force smartish option
#     # 1 only change bc_temp
#     # 2 loop over t_0
#
#     dt_in_min = 15
#     data = []
#
#     T_0 = T_min
#     duration_min = time_in_h
#     m = 1
#     while m < 4:
#         T0_step = 10.0 ** (2 - m)
#
#
#         T_bc = T_limit
#         n = 1
#         bc_data = []
#
#         best_duration = time_in_h
#         while n < 4:
#             t_step = 10.0 ** (2 - n)
#
#             variables_2 = [T_bc,T_0]
#             duration = tempeature_optimization_2(variables_2, T_limit, DOH_min, dt_in_min, time_in_h)
#
#             bc_data.append([T_bc,duration])
#
#             # good computation
#             if duration < time_in_h:
#                 n += 1
#                 inner_data = [T_bc, T_0, duration]
#                 best_duration = duration
#                 T_bc += t_step - t_step/10
#             else:
#                 T_bc -= t_step
#             if T_bc < T_min:
#                 print('Min temp reached!!!')
#                 break
#
#                 #     if T_limit_reached:
#                 #         n += 1
#                 #         bc_temp -= t_increae - t_increae/10
#                 #     else:
#                 #         data.append([bc_temp,T_max,DOH_min,duration])
#                 #         bc_temp += t_increae
#
#         # compare computed duration to last one
#         # if better o
#         if best_duration <= duration_min:
#             duration_min = best_duration
#             T_0 += T0_step
#             # duration is worse
#             # TODO get values from inner data, as theses are the correct temperature values!
#             optimum = [T_bc,T_0,duration_min]
#         else:
#             m += 1
#             T_0 -= T0_step - T0_step/10
#
#
#
#
#         data.append(inner_data)
#
#     for values in data:
#         print(values)
#
#     print('optimum:', optimum)


#     # optimium is somewhere close to t_bc = 49.0 and t0 = 20.2 with a duration of 20.5 hours
# elif setting == 4:
#
#     def test_fkt(x):
#         z = ((x[0])**2)*((x[1])**2)
#         return z
#     print(test_fkt([.010536512,.010536512]))
#     print(test_fkt([-.010536512,.010536512]))
#     print(test_fkt([.010536512,-.010536512]))
#     print(test_fkt([-.010536512,-.010536512]))
#
#     x0 = [10,10]
#     result = minimize(test_fkt,x0)
#     print(result.x)
#
#



#
#
# #while
# data = []
# T_limit = 70
# DOH_min = 0.8
# #trying to optimize the boundary!
# # very simplified "optimization"
# # max temp = 70
# # max difference = 19
#
# # highest constant boundary temp, without exceeding 70°C
# T_0 = 30
# bc_temp = 20.0 # initial temperature
# n = 1 # exponent
# bc_data = []
# #for bc_temp in [20,30,40,50,60,70]:
# while n < 3 :
#     t_increae = 10.0**(2-n)
#
#
# #parameters = None # using the current default values
#     parameters = concrete_experiment.Parameters()
#     parameters['T_0'] = T_0  # inital concrete temperature
#     parameters['T_bc1'] = bc_temp  # temperature boundary value 1
#
#     experiment = concrete_experiment.get_experiment('ConcreteCube',parameters)
#     filename = f'output_T_bc_{bc_temp}'
#     problem = concrete_problem.ConcreteThermoMechanical(experiment,parameters,pv_name = filename)
#
#     min_DOH_sensor = concrete_experiment.MinDOHSensor()
#     max_T_sensor = concrete_experiment.MaxTemperatureSensor()
#
#     problem.add_sensor(min_DOH_sensor)
#     problem.add_sensor(max_T_sensor)
#
#
#     # data for time stepping
#     #time steps
#     dt = 60*60 # time step
#     hours = 48
#     time = hours*60*60         # total simulation time in s
#     # set time step
#     problem.set_timestep(dt) # for time integration scheme
#
#
#     #initialize time
#     t = dt # first time step time
#
#     T_limit_reached = False
#     DOH_min_reached = False
#     while t <= time:
#
#         print('time =', t)
#         # solve temp-hydration-mechanics
#         problem.solve(t=t) # solving this
#
#         # plot fields
#         problem.pv_plot(t=t)
#
#         #print(problem(u_sensor))
#         # prepare next timestep
#         t += dt
#
#         # check last max temp value
#         if problem.sensors[1].data[-1][1] > T_limit:
#             print(f'Max temperature exceeded! Stopping computation at t={t-dt}.')
#             T_limit_reached = True
#             break
#         # check for DOH limit
#         if problem.sensors[0].data[-1][1] > DOH_min:
#             print(f'Min DOH requirement reached! Stopping computation at t={t-dt}.')
#             DOH_min_reached = True
#             break
#
#     # max temperature
#     problem_data = np.asmatrix(problem.sensors[1].data)
#     T_data = problem_data[:, 1]
#     T_max = np.amax(T_data)
#     # min DOH
#     DOH_min = problem.sensors[0].data[-1][1]
#     duration = problem.sensors[0].data[-1][0]
#
#     if T_limit_reached:
#         n += 1
#         bc_temp -= t_increae - t_increae/10
#     else:
#         data.append([bc_temp,T_max,DOH_min,duration])
#         bc_temp += t_increae
#
#
#
#
# for set in data:
#     print(set)