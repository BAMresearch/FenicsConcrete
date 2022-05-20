import fenics_concrete
import dolfin as df
import os

import pytest

def setup_test(parameters,sensor):
    experiment = fenics_concrete.ConcreteCubeUniaxialExperiment(parameters)

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    problem = fenics_concrete.ConcreteThixMechanical(experiment, parameters, pv_name=file_path + 'test_displ_thix')
    problem.experiment.apply_displ_load(parameters['u_bc'])
    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])
    # problem.add_sensor(sensor)
    # data for time stepping
    dt = 0.5 * 60  # 0.5 min step
    time = 30 * 60  # total simulation time in s
    # time = 60

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    age0 = df.Expression('age_zero', age_zero=parameters['age_zero'], degree=0)
    problem.set_initial_age(age0)

    # initialize time
    t = 0
    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

    # computed stress at the end
    print('stresses',problem.sensors[sensor[0].name].data[-1])
    print('strains',problem.sensors[sensor[1].name].data[-1])
    strains = problem.sensors[sensor[1].name].data[-1]
    sensor_stress_max = problem.sensors[sensor[0].name].data[-1].max()
    print(sensor_stress_max)
    # expected stress value
    age_end = time + parameters['age_zero']
    E_end = problem.p.E_0 + problem.p.R_E * problem.p.t_f + problem.p.A_E * (age_end - problem.p.t_f)
    print('E_end',E_end)
    print(parameters['u_bc'] / 1 * E_end)

    # #2D case plane stress:
    # print(strains[0], strains[-1])
    # print(E_end/(1-parameters["nu"]**2)*(parameters["nu"]*strains[0]+strains[-1]))

    assert sensor_stress_max == pytest.approx(
        parameters['u_bc'] / 1 * E_end)  # compare computed stress with the E*strain

def test_displ_thix_3D():
    '''
        uniaxial tension test displacement control to check thixotropy material class
    '''
    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 3 #2
    parameters['mesh_density'] = 2
    parameters['log_level'] = 'INFO'
    parameters['density'] = 0.0
    parameters['u_bc'] = 0.1
    parameters['bc_setting'] = 'disp'
    parameters['age_zero'] = 10 #s
    parameters['nu'] = 0.2

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.5, 1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.5, 1))

    setup_test(parameters,[sensor01,sensor02])

def test_displ_thix_2D():
    '''
        uniaxial tension test displacement control to check thixotropy material class
    '''
    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 2
    parameters['mesh_density'] = 5
    parameters['log_level'] = 'INFO'
    parameters['density'] = 0.0
    parameters['u_bc'] = 0.1
    parameters['bc_setting'] = 'disp'
    parameters['age_zero'] = 10 #s
    parameters['nu'] = 0.2

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5,0.5))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,0.5))

    setup_test(parameters,[sensor01,sensor02])

if __name__ == '__main__':


    test_displ_thix_3D()

    input()

    test_displ_thix_2D()



