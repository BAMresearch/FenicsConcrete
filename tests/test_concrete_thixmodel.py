import fenics_concrete
import dolfin as df
import os
import numpy as np

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
    E_o_time=[]
    # define load increments
    dfs = np.zeros(int(parameters['time'] / parameters['dt']) + 1)
    dfs[0] = 1
    i = 0
    t = 0 # initialize time
    #solve
    while t <= prop3D.p.time:  # time
        # set load increment
        prop3D.experiment.apply_displ_load(dfs[i]*parameters['u_bc'])
        i += 1

        # solve
        prop3D.solve(t=t)  # solving this
        prop3D.pv_plot(t=t)

        # store Young's modulus
        if t + parameters['age_0'] <= prop3D.p.t_f:
            E_o_time.append(prop3D.p.E_0 + prop3D.p.R_E * (t + parameters['age_0']))
        else:
            E_o_time.append(
                prop3D.p.E_0 + prop3D.p.R_E * prop3D.p.t_f + prop3D.p.A_E * (t + parameters['age_0'] - prop3D.p.t_f))

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
    assert sensor_stress_zz == pytest.approx(
        parameters['u_bc'] / 1 * E_o_time[-1])  # compare computed stress with the E*strain

    # check evaluation of stress = Emodul
    derived_E = np.array(prop3D.sensors[sensor01.name].data)[:, -1] / np.array(prop3D.sensors[sensor02.name].data)[:,
                                                                      -1]
    assert derived_E == pytest.approx(E_o_time)


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

    E_o_time = []
    # define load increments
    dfs = np.zeros(int(parameters['time'] / parameters['dt']) + 1)
    dfs[0] = 1
    i = 0
    t = 0 # initialize time
    #solve
    while t <= prop2D.p.time:  # time
        # set load increment
        prop2D.experiment.apply_displ_load(dfs[i]*parameters['u_bc'])
        i += 1
        # solve
        prop2D.solve(t=t)  # solving this
        prop2D.pv_plot(t=t)

        # store Young's modulus
        if t+parameters['age_0'] <= prop2D.p.t_f:
            E_o_time.append(prop2D.p.E_0 + prop2D.p.R_E * (t+parameters['age_0']))
        else:
            E_o_time.append(prop2D.p.E_0 + prop2D.p.R_E * prop2D.p.t_f + prop2D.p.A_E * (t+parameters['age_0'] - prop2D.p.t_f))

        # prepare next timestep
        t += prop2D.p.dt

    # tests
    # get stresses and strains at the end
    # print('stresses yy', np.array(prop2D.sensors[sensor01.name].data)[:,-1])
    # print('strains yy', np.array(prop2D.sensors[sensor02.name].data)[:,-1])
    strain_T = prop2D.sensors[sensor02.name].data[-1]
    strain_yy = strain_T[-1]
    strain_xx = strain_T[0]

    # print('E_o_time',E_o_time)

    assert strain_yy == pytest.approx(prop2D.p.u_bc) # L==1!
    assert strain_xx == pytest.approx(-prop2D.p.nu*prop2D.p.u_bc)

    sensor_stress_yy = prop2D.sensors[sensor01.name].data[-1][-1] # yy or zz direction depending on problem dimension
    # expected stress value
    assert sensor_stress_yy == pytest.approx(
        parameters['u_bc'] / 1 * E_o_time[-1])  # compare computed stress with the E*strain

    # check evaluation of stress = Emodul
    derived_E = np.array(prop2D.sensors[sensor01.name].data)[:,-1]/np.array(prop2D.sensors[sensor02.name].data)[:,-1]
    assert derived_E == pytest.approx(E_o_time)

@pytest.mark.parametrize("R_E", [0,10e4])
def test_density_thix_2D(R_E):
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
    parameters['R_E'] = R_E # if 0 no change in time!
    parameters['A_E'] = 0
    parameters['t_f'] = 4 * 60 # > time -> will not reached!

    parameters['time'] = 2 * 60  # total simulation time in s
    parameters['dt'] = 1 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,0.0)) #1.strainsensor middle bottom
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5,0.5)) #2.strainsensor middle middle
    sensor03 = fenics_concrete.sensors.ReactionForceSensorBottom()
    sensor04 = fenics_concrete.sensors.StressSensor(df.Point(0.5,0.0))
    sensor05 = fenics_concrete.sensors.DisplacementSensor(df.Point(0.5,1.0)) # middle top

    prop2D = setup_test(parameters,[sensor01,sensor02,sensor03,sensor04,sensor05])

    #solve
    E_o_time = []
    # initialize time
    t = 0
    dfs = np.zeros(int(parameters['time']/parameters['dt'])+1)
    dfs[0] = 1
    i = 0
    while t <= prop2D.p.time:  # time
        # set load increment
        prop2D.df.assign(dfs[i])
        i += 1
        # solve
        prop2D.solve(t=t)  # solving this
        prop2D.pv_plot(t=t)

        # store Young's modulus
        if t + parameters['age_0'] <= prop2D.p.t_f:
            E_o_time.append(prop2D.p.E_0 + prop2D.p.R_E * (t + parameters['age_0']))
        else:
            E_o_time.append(
                prop2D.p.E_0 + prop2D.p.R_E * prop2D.p.t_f + prop2D.p.A_E * (t + parameters['age_0'] - prop2D.p.t_f))

        # prepare next timestep
        t += prop2D.p.dt


    # output over time steps in yy direction
    # print('E_o_time', E_o_time)
    # print('sig_o_time', np.array(prop2D.sensors[sensor04.name].data)[:,-1])
    # print('eps_o_time', np.array(prop2D.sensors[sensor01.name].data)[:,-1])
    # print('disp_o_time', np.array(prop2D.sensors[sensor05.name].data)[:,-1])
    # print('force_o_time', prop2D.sensors[sensor03.name].data)

    # tests
    strain_bottom_0 = prop2D.sensors[sensor01.name].data[0][-1] # eps_yy at the start
    strain_bottom_end = prop2D.sensors[sensor01.name].data[-1][-1] # eps_yy at the end
    force_bottom = np.sum(prop2D.sensors[sensor03.name].data) # sum of all force values
    stress_bottom_end = prop2D.sensors[sensor04.name].data[-1][-1] # eps_yy at the end

    # print('strain analytic t=0', -parameters['density']*prop2D.p.g/E_o_time[0])
    # print('dead load', -parameters['density']*prop2D.p.g*1*1)
    # print('force_bottom', force_bottom)
    # print('displacement', prop2D.displacement((0.5,0.5)))

    # standard
    assert force_bottom == pytest.approx(-parameters['density']*prop2D.p.g*1*1) # dead load of full structure
    assert strain_bottom_0 == pytest.approx(-parameters['density']*prop2D.p.g/E_o_time[0], abs=1e-4)

    # evolution of strain
    assert strain_bottom_0 == pytest.approx(strain_bottom_end, abs=1e-8) # if load is applied immediately

    # check if stress changes accordingly to change in E_modul (for last two values!)
    stress_end_prognosis = E_o_time[-1] / E_o_time[-2] * prop2D.sensors[sensor04.name].data[-2][-1]
    # print('test stress end', stress_end_prognosis)
    assert stress_bottom_end == pytest.approx(stress_end_prognosis, abs=1e-8)  # if load is applied immediately


# if __name__ == '__main__':
#
#
#     # test_displ_thix_2D()
#
#     # test_displ_thix_3D()
#     #
#     # test_density_thix_2D(0)
#     # test_density_thix_2D(10e4)



