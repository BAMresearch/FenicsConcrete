import numpy as np

import fenics_concrete

import pytest

# import warnings
# from ffc.quadrature.deprecation \
#     import QuadratureRepresentationDeprecationWarning
# warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

def test_simulation():

    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 2
    parameters['mesh_density'] = 2
    parameters['log_level'] = 'WARNING'

    experiment = fenics_concrete.ConcreteCubeExperiment(parameters)

    problem = fenics_concrete.ConcreteThixMechanical(experiment, parameters)

    # data for time stepping
    dt = 0.5*60  # 0.5 min step
    time = 30*60  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    # initialize time
    t = dt  # first time step time ??? == 0 ???

    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this

        # prepare next timestep
        t += dt

if __name__ == '__main__':


    test_simulation()



