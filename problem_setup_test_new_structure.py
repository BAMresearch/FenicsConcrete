
import time as timer

import concrete_model
#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------
start = timer.time()

parameters = concrete_model.Parameters() # using the current default values
# boundary values...
parameters['dim'] = 2 # inital concrete temperature
parameters['mesh_density'] = 10  # inital concrete temperature
#parameters['mesh_setting'] = 'left/right'  # inital concrete temperature
parameters['degree'] = 2 # inital concrete temperature
parameters['bc_setting'] = 'test-setup'  # default boundary setting
parameters['T_0'] = 10  # inital concrete temperature
parameters['T_bc1'] = 20  # temperature boundary value 1
parameters['T_bc2'] = 30  # temperature boundary value 2

experiment = concrete_model.get_experiment('ConcreteCube',parameters)
problem = concrete_model.ConcreteThermoMechanical(experiment,parameters)

# testing
dohhom_sensor = concrete_model.sensors.MinDOHSensor()
u_sensor = concrete_model.sensors.DisplacementSensor((0.5,0.5))
doh_sensor = concrete_model.sensors.DOHSensor((0.5,0.5))
problem.add_sensor(dohhom_sensor)
problem.add_sensor(u_sensor)
problem.add_sensor(doh_sensor)



# data for time stepping
#time steps
dt = 3600 # time step
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

for sensor in problem.sensors:
    print(sensor.data)
