'''
    multiple layer geometry in 2D - plane strain
    - thix concrete model (FenicsConcrete)
    - pseudo density approach for activation
'''

import fenics_concrete
import dolfin as df
import os
import numpy as np

import pytest

def set_test_parameters():
    # define global parameters used in all layer tests:

    parameters = fenics_concrete.Parameters()  # using the current default values

    parameters['dim'] = 2
    parameters['stress_state'] = 'plane_strain'
    parameters['degree'] = 1
    parameters['layer_number'] = 5 # changed in single layer test!!
    parameters['layer_height'] = 1 / 100 # m
    parameters['layer_width'] = 4 / 100  # m
    parameters['log_level'] = 'INFO'
    parameters['density'] = 2070. # kg/m³
    # parameters['g'] = 9.81 # in material_problem.py default value

    # material parameters from Wolfs et al 2018
    parameters['nu'] = 0.3  # Poissons Ratio see Wolfs et al 2018
    parameters['E_0'] = 0.078e6  # Youngs Modulus Pa
    parameters['R_E'] = 0.0  # reflocculation rate of E modulus in Pa / s
    parameters['A_E'] = 0.0012e6/60  # structuration rate of E modulus in Pa / s
    parameters['t_f'] = 0  # reflocculation time in s
    parameters['age_0'] = 0 # s concrete age at print head

    parameters['t_layer'] = 20 # s time to build one layer # Wolf paper 0.31 min!
    parameters['dt'] = 5 # time step

    return parameters

def setup_problem(parameters, pv_name):
    # define problem
    experiment = fenics_concrete.MultipleLayers2DExperiment(parameters)
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    problem = fenics_concrete.ConcreteThixMechanical(experiment, parameters, pv_name=file_path + pv_name)

    # sensor
    problem.add_sensor(fenics_concrete.sensors.ReactionForceSensorBottom())
    problem.add_sensor(fenics_concrete.sensors.StrainSensor(df.Point(parameters['layer_width']/2,0.0)))
    problem.add_sensor(fenics_concrete.sensors.StressSensor(df.Point(parameters['layer_width']/2,0.0)))

    return problem


def define_path_time(prob,param,t_diff,t_0=0):
    # create path as layer wise and overwrite in prob
    # one layer by time
    '''
    prob: problem
    param: parameter dictionary
    t_diff: time difference between each layer
    t_0: start time for all (0 if static computation)
                            (-end_time last layer if dynamic computation)
    '''

    # extract default fct to be replaced
    tmp_path = prob.mechanics_problem.q_path
    # print(tmp_age.vector()[:])
    # dof map for coordinates
    dof_map = tmp_path.function_space().tabulate_dof_coordinates()[:]
    # print(tmp_age.function_space().dofmap().tabulate_all_coordinates())
    new_path = np.zeros(len(tmp_path.vector()[:]))
    y_CO = np.array(dof_map)[:, -1]
    h_min = np.arange(0, param['layer_number'] * param['layer_height'], param['layer_height'])
    h_max = np.arange(param['layer_height'], (param['layer_number'] + 1) * param['layer_height'],
                      param['layer_height'])
    # print(y_CO)
    # print(h_min)
    # print(h_max)
    for i in range(0, len(h_min)):
        layer_index = np.where((y_CO > h_min[i] - df.DOLFIN_EPS) & (y_CO <= h_max[i] + df.DOLFIN_EPS))
        # print((parameters['layer_number']-i-1)*age_diff_layer)
        new_path[layer_index] = t_0 + (param['layer_number'] - 1 - i) * t_diff
    # print('new_path', new_path, new_path.min(), new_path.max())

    prob.mechanics_problem.q_path.vector()[:] = new_path[:]  # overwrite

    return prob

def test_single_layer_2D_CS():
    # crossection like "1m" length
    # One single layer for a given time

    # set parameters
    parameters = set_test_parameters()
    parameters['layer_number'] = 1
    parameters['age_0'] = 20
    parameters['degree'] = 2

    # set standard problem & sensor
    pv_name = 'test_single_layer_thix'
    problem = setup_problem(parameters,pv_name)

    # set time step
    dt = parameters['dt']
    time = parameters['t_layer']  # total simulation time in s: time to build one layer based on Wolfs paper 0.31 min !

    problem.set_timestep(dt)  # for time integration scheme
    # initialize time
    t = 0
    while t <= time:  # time
        # solve temp-hydration-mechanics
        # print('solve for', t)
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

    # check results (multi-axial stress state not uniaxial no analytical stress solution)
    force_bottom = problem.sensors["ReactionForceSensorBottom"].data[-1]
    force_structure = parameters['density']*parameters['layer_width']*parameters['layer_height']*problem.p.g
    # print('force - weigth', force_bottom, force_structure )
    assert force_bottom == pytest.approx(-force_structure) # dead load of full structure

    # print('strains', problem.sensors["StrainSensor"].data[:])
    # print('stresses', problem.sensors["StressSensor"].data[:])
    stress = np.array(problem.sensors["StressSensor"].data[:])
    # print('stress',stress)
    assert np.absolute(np.diff(stress,axis=0)).max() == pytest.approx(0.0) # stress should be constant over time

    strain = np.array(problem.sensors["StrainSensor"].data[:])
    dstrain = np.diff(strain,axis=0) # should be constant since stiffness increases constant in time
    # print('strain',strain)
    # print('diff_strain',dstrain)
    assert np.absolute(np.diff(dstrain,axis=0)).max() == pytest.approx(0.0,abs=1e-8) # strain increment should be constant (reziproc related to R_E)


def test_multiple_layer_2D_CS_static():
    # serveral layers with different age given
    # static computation no time loop

    parameters = set_test_parameters()

    # set standard problem & sensor
    pv_name = 'test_multilayer_thix_static'
    problem = setup_problem(parameters, pv_name)

    # create layers path time for current static time (== t=80 in dynamic case)
    path=df.Expression('0', degree=0)
    problem.set_initial_path(path)
    problem = define_path_time(problem, parameters, parameters['t_layer'])

    # initialize time compute just one time step of current situation (path!)
    problem.set_timestep(0)
    t = 0
    # print('solve for', t)
    problem.solve(t=t)  # solving this
    problem.pv_plot(t=t)

    # check global solution eigenvalue = bottom force
    # print('strain sensor bottom', problem.sensors["StrainSensor"].data[:])
    # print('stress sensor bottom', problem.sensors["StressSensor"].data[:])
    force_bottom = problem.sensors["ReactionForceSensorBottom"].data[-1]
    force_structure = parameters["layer_number"] * parameters['density'] * parameters['layer_width'] * parameters['layer_height'] * problem.p.g
    # print('force - weigth', force_bottom, force_structure)
    assert force_bottom == pytest.approx(-force_structure)

    # check Young' modulus distribution
    if parameters['t_f'] == 0: # changed thixotropy model CHECK will not work!
        E_bottom_layer = problem.p.E_0 + problem.p.A_E*(parameters['layer_number']-1)*parameters['t_layer']
        E_upper_layer = problem.p.E_0
        assert E_upper_layer == pytest.approx(problem.mechanics_problem.q_E.vector()[:].min())
        assert E_bottom_layer == pytest.approx(problem.mechanics_problem.q_E.vector()[:].max())
        #TODO: Emodulus sensor?


    return problem.sensors["StrainSensor"].data[:], problem.sensors["StressSensor"].data[:]

def test_multiple_layer_2D_CS_dynamic():
    # serveral layers dynamically deposited path given

    parameters = set_test_parameters()

    # set standard problem & sensor
    pv_name = 'test_multilayer_thix_dynamic'
    problem = setup_problem(parameters, pv_name)

   # Layers given by path function
    path=df.Expression('0', degree=0)
    problem.set_initial_path(path)
    time_last_layer_set = (parameters['layer_number']-1) * parameters['t_layer']
    problem = define_path_time(problem, parameters, parameters['t_layer'], t_0=-time_last_layer_set)

    # initialize time
    dt = parameters['dt']
    time = (parameters['layer_number']-1) * parameters['t_layer']  # total simulation time in s
    problem.set_timestep(dt)
    t = 0
    while t <= time:  # time
        # solve temp-hydration-mechanics
        # print('solve for', t)
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

    # check residual force bottom
    force_bottom = problem.sensors["ReactionForceSensorBottom"].data[-1]
    force_structure = parameters["layer_number"] * parameters['density'] * parameters['layer_width'] * parameters['layer_height'] * problem.p.g
    # print('force - weigth', force_bottom, force_structure)
    assert force_bottom == pytest.approx(-force_structure)

    # check result with static result
    strain_static, stress_static = test_multiple_layer_2D_CS_static()
    time_line = np.linspace(0, time, int(time / dt + 1))
    time_ind = np.where(time_line==(parameters['layer_number']-1)*parameters['t_layer'])[0][0]
    # print('static',strain_static[-1], stress_static[-1])
    # print('dynamic',problem.sensors["StrainSensor"].data[time_ind], problem.sensors["StressSensor"].data[time_ind])
    assert strain_static[-1] == pytest.approx(problem.sensors["StrainSensor"].data[time_ind])
    assert stress_static[-1] == pytest.approx(problem.sensors["StressSensor"].data[time_ind])

    # # strain_yy over time
    # import matplotlib.pylab as plt
    # print(np.array(problem.sensors["StrainSensor"].data[:])[:,-1])
    # plt.figure(1)
    # plt.plot(time_line,np.array(problem.sensors["StrainSensor"].data[:])[:,-1],'*-r')
    # plt.xlabel('process time')
    # plt.ylabel('eps_yy bottom middle')
    # plt.show()




#
# if __name__ == '__main__':


    # test_single_layer_2D_CS()

    # test_multiple_layer_2D_CS_static()
    # #
    # test_multiple_layer_2D_CS_dynamic()