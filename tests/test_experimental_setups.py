import numpy as np

import fenics_concrete

import pytest

# import warnings
# from ffc.quadrature.deprecation \
#     import QuadratureRepresentationDeprecationWarning
# warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

def simple_simulation(parameters,experiment):

    problem = fenics_concrete.ConcreteThermoMechanical(experiment, parameters)

    # data for time stepping
    dt = 1200  # 20 min step
    time = dt * 3  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    # initialize time
    t = dt  # first time step time

    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this

        # prepare next timestep
        t += dt


# testing the different experimental setup with options
# just checking that they run!


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("get_experiment", [fenics_concrete.ConcreteCubeExperiment,
                                            fenics_concrete.MinimalCubeExperiment,
                                            fenics_concrete.ConcreteColumnExperiment,
                                            fenics_concrete.ConcreteBeamExperiment,
                                            ])
def test_experiemental_setup(dim, get_experiment):

    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = dim
    parameters['mesh_density'] = 2
    parameters['log_level'] = 'WARNING'

    experiment = get_experiment(parameters)

    simple_simulation(parameters, experiment)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("boundary_setting", ['free', 'fixed'])
def test_cylinder_setup(dim, boundary_setting):

    parameters = fenics_concrete.Parameters()  # using the current default values

    parameters['mesh_density'] = 4
    parameters['log_level'] = 'WARNING'
    parameters['E'] = 3000
    parameters['nu'] = 0.2
    parameters['radius'] = 75
    parameters['height'] = 300
    parameters['bc_setting'] = boundary_setting
    parameters['dim'] = dim

    displacement = -parameters.height * 0.1

    experiment = fenics_concrete.ConcreteCylinderExperiment(parameters)
    problem = fenics_concrete.LinearElasticity(experiment, parameters, vmapoutput=True)

    problem.experiment.apply_displ_load(displacement)

    problem.solve()  # solving this

