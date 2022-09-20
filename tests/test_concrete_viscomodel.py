import fenics_concrete
import dolfin as df
import os
import numpy as np

import pytest

def setup_test(parameters,sensor):
    experiment = fenics_concrete.ConcreteCubeUniaxialExperiment(parameters)

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    problem = fenics_concrete.ConcreteAMMechanical(experiment, parameters, mech_prob_string='ConcreteViscoElasticModel', pv_name=file_path + 'test_visco')
    if parameters['bc_setting'] == 'disp':
        problem.experiment.apply_displ_load(parameters['u_bc'])
    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])
    # problem.add_sensor(sensor)

    # set time step
    problem.set_timestep(problem.p.dt)  # for time integration scheme

    return problem

def test_relaxation_2D():
    '''
        uniaxial tension test displacement control to check relaxation of visco material class
    '''
    parameters = fenics_concrete.Parameters() # using the current default values

    parameters['dim'] = 2
    parameters['mesh_density'] = 1
    parameters['log_level'] = 'INFO'

    parameters['density'] = 0.0
    parameters['u_bc'] = 0.002 # == strain since dimensions 1!!
    parameters['bc_setting'] = 'disp'

    parameters['E_0'] = 70e3
    parameters['E_1'] = 20e3
    parameters['eta'] = 2e3     # relaxation time: tau = eta/E_1
    parameters['nu'] = 0.2
    parameters['stress_state'] = 'plane_strain'

    parameters['time'] = 1  # total simulation time in s
    parameters['dt'] = 0.01  # step (should be < tau=eta/E_1)

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(1.0,1.0))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(1.0,1.0))

    prop2D = setup_test(parameters,[sensor01,sensor02])

    eps_o_time = []
    sig_o_time = []
    time = []
    # initialize time and solve!
    t = 0
    while t <= prop2D.p.time:  # time
        time.append(t)
        # solve temp-hydration-mechanics
        prop2D.solve(t=t)  # solving this
        prop2D.pv_plot(t=t)
        # prepare next timestep
        t += prop2D.p.dt


        sig_o_time.append(prop2D.sensors[sensor01.name].data[-1][1]) # sig_yy
        # eps_o_time.append(prop2D.sensors[sensor02.name].data[-1][1])  # eps_yy


    # check with analytic 1D solution
    sig_yy=[]
    tau = parameters['eta']/parameters['E_1']
    eps_r = parameters['u_bc'] # L==1 -> u_bc = eps_r (prescriped strain)
    for i in time:
        sig_yy.append(parameters['E_0']*eps_r + parameters['E_1']*eps_r*np.exp(-i/tau))

    # print('analytic 1D == 2D with nu=0', sig_yy)
    # print('stress over time', sig_o_time)
    error2ana = np.linalg.norm(sig_yy-np.array(sig_o_time))/np.linalg.norm(sig_yy)
    # print('norm error sig_yy', error2ana)

    assert error2ana < 5e-3

    # import matplotlib.pyplot as plt
    #
    # plt.plot(time, sig_yy, '*r', label='analytic')
    # plt.plot(time, sig_o_time,'og', label='FEM')
    # plt.legend()
    # plt.show()

    # get stresses and strains at the end
    # print('stresses',prop2D.sensors[sensor01.name].data[-1])
    # print('strains',prop2D.sensors[sensor02.name].data[-1])
    strain_xx = prop2D.sensors[sensor02.name].data[-1][0]
    strain_yy = prop2D.sensors[sensor02.name].data[-1][1]
    assert strain_yy == pytest.approx(prop2D.p.u_bc) # L==1!
    assert strain_xx == pytest.approx(-prop2D.p.nu*prop2D.p.u_bc)


#TODO: add creep test requires load boundary condition not yet in the experiment framework

# if __name__ == '__main__':
#
#
#     test_relaxation_2D()





