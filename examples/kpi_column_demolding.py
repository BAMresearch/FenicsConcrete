import concrete_model
import numpy as np


def kpi_column_demoulding(T_0, T_bc):

    #----------1------------------
    # model paramters
    parameters = concrete_model.Parameters() # using the current default values
    # output
    parameters['log_level'] = 'WARNING'
    # mesh parameters
    parameters['dim'] = 2 # dimension of problem 2D/3D
    parameters['mesh_density'] = 10  # number of elements on a length of "1"
    parameters['mesh_density_min'] = 5  # minimal number of elements in spacial direction
    # geometry
    parameters['width'] = 1  # width (square crossection) in m
    parameters['height'] = 4 # height in m

    # temperature conditions
    parameters['T_0'] = T_0  # inital concrete temperature
    parameters['T_bc1'] = T_bc  # temperature boundary value

    timestep = 15 # in minutes
    max_time = 48 # in hours


    #------------- 2 -----------------------------------------
    # problem setup
    experiment = concrete_model.ConcreteColumnExperiment(parameters)
    problem = concrete_model.ConcreteThermoMechanical(experiment,parameters)

    # sensors
    test = concrete_model.sensors.MaxTemperatureSensor()

    problem.add_sensor(concrete_model.sensors.MaxTemperatureSensor())
    problem.add_sensor(concrete_model.sensors.MaxYieldSensor())

    #--------------- 3 ---------------------------------------
    # data for time stepping
    dt = 60*timestep      # time step in seconds
    time = max_time*60*60 # maximum simulation time in seconds

    # set time step
    problem.set_timestep(dt) # for time integration scheme

    #initialize time
    t = dt # first time step time


    check_yield = False
    check_max_temp = False

    # loop till both check yield and check max temp are reached OR time limit
    while t <= time and (not check_yield or not check_max_temp): # time

        # solve temp-hydration-mechanics
        problem.solve(t=t) # solving this

        # check yield, once reached, save the time
        if problem.sensors.MaxYieldSensor.data[-1] < 0 and not check_yield:
            check_yield = True
            yield_time = t

        # assuming that once the max temp is decreasing the maximum has been reached
        # TODO: this might be to simple...
        if problem.sensors.MaxTemperatureSensor.data[-1] < problem.sensors.MaxTemperatureSensor.max:
            check_max_temp = True

        # prepare next timestep
        t += dt


    # interpolate demould time and max temp
    if  check_yield:
        #interpolate demoulding time, using date from last two steps
        yield_index = problem.sensors.MaxYieldSensor.time.index(yield_time)
        yield_2 = problem.sensors.MaxYieldSensor.data[yield_index]
        yield_1 = problem.sensors.MaxYieldSensor.data[yield_index-1]
        time_2 = problem.sensors.MaxYieldSensor.time[yield_index]
        time_1 = problem.sensors.MaxYieldSensor.time[yield_index-1]
        demoulding_time = -yield_1*(time_2-time_1)/(yield_2-yield_1)+ time_1
        demoulding_time = demoulding_time/60/60 # in hours
    else:
        print(' * Warning: Yield has not been reached, increase max time!')
        demoulding_time = None


    if check_max_temp:
        max_temperature = problem.sensors.MaxTemperatureSensor.max
    else:
        print(' * Warning: Maximum temperature has not been reached, increase max time!')
        max_temperature = None

    return demoulding_time, max_temperature


# testing interface
T_0 = 29.0
T_bc = 36.0

demoulding_time, max_temperature = kpi_column_demoulding(T_0, T_bc)
print(T_0,T_bc,demoulding_time,max_temperature)

# optimun around (29,36)





