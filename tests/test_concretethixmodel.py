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

    # set time step
    problem.set_timestep(problem.p.dt)  # for time integration scheme

    age0 = df.Expression('age_zero', age_zero=parameters['age_zero'], degree=0)
    problem.set_initial_age(age0)

    # initialize time
    t = 0
    while t <= problem.p.time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += problem.p.dt

    return problem

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

    parameters['time'] = 30 * 60  # total simulation time in s
    parameters['dt'] = 0.5 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.5, 1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.5, 1))

    prop3D = setup_test(parameters,[sensor01,sensor02])

    # tests
    # get stresses and strains at the end
    # print('stresses',prop3D.sensors[sensor01.name].data[-1])
    # print('strains',prop3D.sensors[sensor02.name].data[-1])
    strain_T = prop3D.sensors[sensor02.name].data[-1]
    strain_zz = strain_T[-1]
    strain_xx = strain_T[0]
    strain_yy = strain_T[4]

    assert strain_zz == pytest.approx(prop3D.p.u_bc)  # L==1!
    assert strain_xx == pytest.approx(-prop3D.p.nu * prop3D.p.u_bc)
    assert strain_yy == pytest.approx(-prop3D.p.nu * prop3D.p.u_bc)

    sensor_stress_zz = prop3D.sensors[sensor01.name].data[
        -1][-1]
    # expected stress value
    age_end = parameters['time'] + parameters['age_zero']
    E_end = prop3D.p.E_0 + prop3D.p.R_E * prop3D.p.t_f + prop3D.p.A_E * (age_end - prop3D.p.t_f)
    assert sensor_stress_zz == pytest.approx(
        parameters['u_bc'] / 1 * E_end)  # compare computed stress with the E*strain


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
    parameters['stress_state'] = 'plane_stress'

    parameters['time'] = 30 * 60  # total simulation time in s
    parameters['dt'] = 0.5 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5,1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,1))

    prop2D = setup_test(parameters,[sensor01,sensor02])

    # tests
    # get stresses and strains at the end
    # print('stresses',prop2D.sensors[sensor01.name].data[-1])
    # print('strains',prop2D.sensors[sensor02.name].data[-1])
    strain_T = prop2D.sensors[sensor02.name].data[-1]
    strain_yy = strain_T[-1]
    strain_xx = strain_T[0]

    assert strain_yy == pytest.approx(prop2D.p.u_bc) # L==1!
    assert strain_xx == pytest.approx(-prop2D.p.nu*prop2D.p.u_bc)

    sensor_stress_yy = prop2D.sensors[sensor01.name].data[-1][-1] # yy or zz direction depending on problem dimension
    # expected stress value
    age_end = parameters['time'] + parameters['age_zero']
    E_end = prop2D.p.E_0 + prop2D.p.R_E * prop2D.p.t_f + prop2D.p.A_E * (age_end - prop2D.p.t_f)
    assert sensor_stress_yy == pytest.approx(
        parameters['u_bc'] / 1 * E_end)  # compare computed stress with the E*strain


# if __name__ == '__main__':
#
#
#     test_displ_thix_2D()
#
#     test_displ_thix_2D()



