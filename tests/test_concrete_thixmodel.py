import fenics_concrete
import dolfin as df
import os

import pytest

def setup_test(parameters,sensor):
    experiment = fenics_concrete.ConcreteCubeUniaxialExperiment(parameters)

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    problem = fenics_concrete.ConcreteAMMechanical(experiment, parameters, mech_prob_string='ConcreteThixElasticModel', pv_name=file_path + 'test_thix')
    if parameters['bc_setting'] == 'disp':
        problem.experiment.apply_displ_load(parameters['u_bc'])
    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])
    # problem.add_sensor(sensor)

    # set time step
    problem.set_timestep(problem.p.dt)  # for time integration scheme

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
    parameters['age_0'] = 10 #s # age of concrete at expriment start time
    parameters['nu'] = 0.2

    parameters['time'] = 30 * 60  # total simulation time in s
    parameters['dt'] = 0.5 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.5, 1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.5, 1))

    prop3D = setup_test(parameters,[sensor01,sensor02])

    #solve
    # initialize time
    t = 0
    while t <= prop3D.p.time:  # time
        # solve temp-hydration-mechanics
        prop3D.solve(t=t)  # solving this
        prop3D.pv_plot(t=t)
        # prepare next timestep
        t += prop3D.p.dt

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
    age_end = parameters['time'] + parameters['age_0']
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
    parameters['age_0'] = 10 #s # age of concrete at expriment start time
    parameters['nu'] = 0.2
    parameters['stress_state'] = 'plane_stress'

    parameters['time'] = 30 * 60  # total simulation time in s
    parameters['dt'] = 0.5 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5,1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,1))

    prop2D = setup_test(parameters,[sensor01,sensor02])

    #solve
    # initialize time
    t = 0
    while t <= prop2D.p.time:  # time
        # solve temp-hydration-mechanics
        prop2D.solve(t=t)  # solving this
        prop2D.pv_plot(t=t)
        # prepare next timestep
        t += prop2D.p.dt

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
    age_end = parameters['time'] + parameters['age_0']
    E_end = prop2D.p.E_0 + prop2D.p.R_E * prop2D.p.t_f + prop2D.p.A_E * (age_end - prop2D.p.t_f)
    assert sensor_stress_yy == pytest.approx(
        parameters['u_bc'] / 1 * E_end)  # compare computed stress with the E*strain


def test_density_thix_2D():
    '''
        uniaxial tension test with density without change in Young's modulus over time
        checking general implementation
    '''
    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 2
    parameters['mesh_density'] = 5
    parameters['degree'] = 2
    parameters['log_level'] = 'INFO'
    parameters['density'] = 2070.0
    parameters['bc_setting'] = 'density'
    parameters['age_0'] = 0 #s # age of concrete at expriment start time
    parameters['nu'] = 0.2
    parameters['stress_state'] = 'plane_stress'

    parameters['E_0'] = 2070000
    parameters['R_E'] = 10e4 # no change in time!
    parameters['A_E'] = 0
    parameters['t_f'] = 300

    parameters['time'] = 2 * 60  # total simulation time in s
    parameters['dt'] = 1 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,0.0)) #1.strainsensor middle bottom
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,0.5)) #2.strainsensor middle middle
    sensor03 = fenics_concrete.sensors.ReactionForceSensorBottom()
    sensor04 = fenics_concrete.sensors.StressSensor(df.Point(0.5,0.0))

    prop2D = setup_test(parameters,[sensor01,sensor02,sensor03,sensor04])

    #solve
    sig_o_time = []
    eps_o_time = []
    E_o_time = []
    # initialize time
    t = 0
    while t <= prop2D.p.time:  # time
        # solve temp-hydration-mechanics
        prop2D.solve(t=t)  # solving this
        prop2D.pv_plot(t=t)

        # sensor output
        if t <= prop2D.p.t_f:
            E_o_time.append(
                prop2D.p.E_0 + prop2D.p.R_E * t)
        eps_o_time.append(prop2D.sensors[list(prop2D.sensors.keys())[0]].data[-1][-1]) # yy
        sig_o_time.append(prop2D.sensors[list(prop2D.sensors.keys())[3]].data[-1][-1]) # yy

        # prepare next timestep
        t += prop2D.p.dt


    print('sig_o_time', sig_o_time)
    print('eps_o_time', eps_o_time)
    print('E_o_time', E_o_time)
    # tests
    # get stresses and strains at the end usually the sensor names are the sensor class name if same kind then numbered!!
    # print(prop2D.sensors.keys()) # available Sensor Names
    # print('strain sensor',prop2D.sensors[list(prop2D.sensors.keys())[0]].data[-1]) # or "StrainSensor" = sensor01.name
    # print('strain sensor', prop2D.sensors[list(prop2D.sensors.keys())[1]].data[-1]) # or "StrainSensor" != sensor02.name !
    strain_bottom = prop2D.sensors[list(prop2D.sensors.keys())[0]].data[-1][-1] # yy
    strain_middle = prop2D.sensors[list(prop2D.sensors.keys())[1]].data[-1][-1] # yy

    # print('reaction force', prop2D.sensors[list(prop2D.sensors.keys())[-1]].data[-1]) # or "ReactionForceSensorBottom" = sensor03.name
    force_bottom = prop2D.sensors[sensor03.name].data[-1]

    print('strain analytic', -parameters['density']*prop2D.p.g/E_o_time[0], -parameters['density']*prop2D.p.g/E_o_time[-1])
    print('strain bottom, strain middle', strain_bottom, strain_middle)
    print('dead load', -parameters['density']*prop2D.p.g*1*1)
    print('force_bottom', force_bottom)
    print('displacement', prop2D.displacement((0.5,0.5)))

    assert force_bottom == pytest.approx(-parameters['density']*prop2D.p.g*1*1) # dead load of full structure
    assert strain_middle == pytest.approx(strain_bottom / 2., abs=1e-4) # linear increase of strain over heigth
    assert strain_bottom == pytest.approx(-parameters['density']*prop2D.p.g/E_o_time[-1], abs=1e-4)



if __name__ == '__main__':


    # test_displ_thix_2D()

    # test_displ_thix_3D()
    #
    test_density_thix_2D()



