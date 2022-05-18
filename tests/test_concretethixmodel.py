import numpy as np

import fenics_concrete
import dolfin as df
import os

import pytest

# import warnings
# from ffc.quadrature.deprecation \
#     import QuadratureRepresentationDeprecationWarning
# warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

def test_displ_thix():

    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 2
    parameters['mesh_density'] = 2
    parameters['log_level'] = 'INFO'
    parameters['density'] = 0.0
    parameters['u_bc'] = 10

    experiment = fenics_concrete.ConcreteCubeExperiment(parameters)
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    problem = fenics_concrete.ConcreteThixMechanical(experiment, parameters, pv_name=file_path+'test_displ_thix')

    # data for time stepping
    dt = 0.5*60  # 0.5 min step
    time = 30*60  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    age0 = df.Expression('age_zero', age_zero=10, degree=0)
    problem.set_initial_age(age0)

    # initialize time
    t = 0

    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

def test_cube_thix():

    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 2
    parameters['mesh_density'] = 2
    parameters['log_level'] = 'INFO'
    parameters['density'] = 2070.
    parameters['u_bc'] = 0.0

    experiment = fenics_concrete.ConcreteCubeExperiment(parameters)
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    problem = fenics_concrete.ConcreteThixMechanical(experiment, parameters, pv_name=file_path+'test_cube_thix')

    # data for time stepping
    dt = 0.5*60  # 0.5 min step
    time = 30*60  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    age0 = df.Expression('age_zero', age_zero=10, degree=0)
    problem.set_initial_age(age0)

    # initialize time
    t = 0

    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

if __name__ == '__main__':


    test_displ_thix()

    test_cube_thix()



