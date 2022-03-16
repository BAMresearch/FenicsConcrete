import numpy as np

import concrete_model

import unittest

# import warnings
# from ffc.quadrature.deprecation \
#     import QuadratureRepresentationDeprecationWarning
# warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

def simple_simulation(parameters,experiment):

    problem = concrete_model.ConcreteThermoMechanical(experiment, parameters)

    # data for time stepping
    dt = 1200  # 20 min step
    time = dt * 3  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    # initialize time
    t = 0  # first time step time

    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this

        # prepare next timestep
        t += dt


# testing the different experimental setup with options
# just checking that they run!

class TestExperimentalSetups(unittest.TestCase):

    def test_concrete_cube_2D(self):

        parameters = concrete_model.Parameters() # using the current default values

        parameters['dim'] = 2
        parameters['mesh_density'] = 2
        parameters['log_level'] = 'WARNING'

        experiment = concrete_model.get_experiment('ConcreteCube',parameters)

        simple_simulation(parameters, experiment)


    def test_concrete_cube_3D(self):

        parameters = concrete_model.Parameters() # using the current default values

        parameters['dim'] = 3
        parameters['mesh_density'] = 2
        parameters['log_level'] = 'WARNING'

        experiment = concrete_model.get_experiment('ConcreteCube',parameters)

        simple_simulation(parameters, experiment)


    def test_minimal_cube_2D(self):

        parameters = concrete_model.Parameters() # using the current default values

        parameters['dim'] = 2
        parameters['mesh_density'] = 2
        parameters['log_level'] = 'WARNING'

        experiment = concrete_model.get_experiment('MinimalCube',parameters)

        simple_simulation(parameters, experiment)


    def test_minimal_cube_3D(self):

        parameters = concrete_model.Parameters() # using the current default values

        parameters['dim'] = 3
        parameters['mesh_density'] = 2
        parameters['log_level'] = 'WARNING'

        experiment = concrete_model.get_experiment('MinimalCube',parameters)

        simple_simulation(parameters, experiment)


    def test_concrete_beam_2D(self):
        parameters = concrete_model.Parameters()  # using the current default values

        parameters['dim'] = 2
        parameters['mesh_density'] = 2
        parameters['log_level'] = 'WARNING'

        experiment = concrete_model.get_experiment('ConcreteBeam', parameters)

        simple_simulation(parameters, experiment)

if __name__ == '__main__':
     unittest.main()