# This is the main exmaple file for solving the linear elasticity problem.
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
import numpy as np
import fenicsX_concrete


def simple_setup(p, sensor):
    parameters = fenicsX_concrete.Parameters()  # using the current default values

    #parameters['log_level'] = 'WARNING'
    parameters['bc_setting'] = 'free'
    parameters['mesh_density'] = 10

    parameters = parameters + p

    experiment = fenicsX_concrete.concreteSlabExperiment(parameters)         # Specifies the domain, discretises it and apply Dirichlet BCs
    #print(help(FConcrete2.ConcreteCylinderExperiment))

    problem = fenicsX_concrete.LinearElasticity(experiment, parameters) #Specidies the material law and weak forms.
    #print(help(fenics_concrete.LinearElasticity))

    problem.add_sensor(sensor)

    #problem.experiment.apply_load_bc(displacement)

    problem.solve()  # solving this

    #print(problem.displacement([0,2]))

    #problem.pv_plot()

    # last measurement
    return problem.sensors[sensor.name].data[-1]


p = fenicsX_concrete.Parameters()  # using the current default values
p['E'] = 1
p['nu'] = 0.2
p['length'] = 1
p['breadth'] = 0.2
p['num_elements_length'] = 20
p['num_elements_breadth'] = 10
#displacement = -3
p['dim'] = 2
sensor = fenicsX_concrete.sensors.DisplacementSensor(np.array([[0.1, 0.123, 0]]))#ReactionForceSensorBottom()
#print(sensor.name)
measured = simple_setup(p, sensor)
print(measured)


