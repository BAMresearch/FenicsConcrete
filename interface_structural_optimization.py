
import time as timer

import concrete_model
import numpy as np
#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------




parameters = concrete_model.Parameters() # using the current default values

parameters['log_level'] = 'WARNING'
# mesh parameters, only relevant if numerical problem arise
parameters['dim'] = 2 # dimension of problem 2D/3D
parameters['mesh_density'] = 10  # number of elements on a length of "1"
parameters['mesh_density_min'] = 5  # minimal number of elements in spacial direction
parameters['T_0'] = 10  # inital concrete temperature
parameters['T_bc1'] = 20  # temperature boundary value

parameters['width'] = 1  # width (square crossection) in m
parameters['height'] = 4 # height in m

experiment = concrete_model.ConcreteColumnExperiment(parameters)
problem = concrete_model.ConcreteThermoMechanical(experiment,parameters)


# testing
maxT_sensor = concrete_model.sensors.MaxTemperatureSensor()
maxYield_sensor = concrete_model.sensors.MaxYieldSensor()

problem.add_sensor(maxT_sensor)
problem.add_sensor(maxYield_sensor)



# data for time stepping
#time steps
dt = 60*20 # time step
hours = 40
time = hours*60*60         # total simulation time in s
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

    # check yield
    if problem.new_sensors.MaxYieldSensor.data[-1] < 0 and not check_yield:
        check_yield = True
        yield_time = t

    if problem.new_sensors.MaxTemperatureSensor.data[-1] < problem.new_sensors.MaxTemperatureSensor.max:
        check_max_temp = True


    print(check_yield, check_max_temp)

    #print(problem(u_sensor))
    # prepare next timestep
    t += dt

print(yield_time)
index = problem.new_sensors.MaxYieldSensor.time.index(yield_time)
print(index)
print(problem.new_sensors.MaxYieldSensor.data[index-1])
print(problem.new_sensors.MaxYieldSensor.data[index])

#interpolate

#TODO interpolate time of formwork removal!!!
#TODO what if yield or temperature not reached in time interval???




#
print(problem.new_sensors.MaxTemperatureSensor.max)
print(problem.new_sensors.MaxTemperatureSensor.data)
print(problem.new_sensors.MaxYieldSensor.max)
print(problem.new_sensors.MaxYieldSensor.data)

# print(problem.sensors[1].data)
# print(problem.sensors[2].data)

# for sensor in problem.sensors:
#     print(sensor.data)

