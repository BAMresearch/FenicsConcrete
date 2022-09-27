import fenics_concrete
import dolfin as df
import os
import numpy as np

import pytest

from fenics_concrete.experimental_setups.experiment import Experiment
from fenics_concrete.helpers import Parameters

class ConcreteTruss(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        p['dim'] = 1  # default boundary setting
        p['mesh_density'] = 10  # default boundary setting
        p = p + parameters
        super().__init__(p)

        # initialize variable top_displacement
        self.displacement = df.Constant(0.0)

    def setup(self):
        # elements per spacial direction
        n = self.p.mesh_density
        if self.p.dim == 1:
            self.mesh = df.IntervalMesh(n, 0, 1.)
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()
    def create_displ_bcs(self, V):

        displ_bcs = []

        displ_bcs.append(df.DirichletBC(V, self.displacement, self.boundary_left()))
        displ_bcs.append(df.DirichletBC(V, df.Constant(0), self.boundary_left()))

        return displ_bcs

    def apply_displ_load(self, displacement):

        self.displacement.assign(df.Constant(displacement))


def setup_test(parameters,sensor):
    experiment = fenics_concrete.ConcreteCubeUniaxialExperiment(parameters)
    # experiment = ConcreteTruss(parameters)
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    # problem = fenics_concrete.ConcreteAMMechanical(experiment, parameters, mech_prob_string='ConcreteViscoElasticModel', pv_name=file_path + 'test_visco')
    problem = fenics_concrete.ConcreteAMMechanical(experiment, parameters, mech_prob_string='ConcreteViscoDevElasticModel',
                                                   pv_name=file_path + 'test_visco')

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
    parameters['mesh_density'] = 2
    parameters['log_level'] = 'INFO'

    parameters['density'] = 0.0
    parameters['u_bc'] = 0.002 # == strain since dimensions 1!!
    parameters['bc_setting'] = 'disp'

    parameters['E_0'] = 70e3
    parameters['E_1'] = 20e3
    parameters['eta'] = 2e3     # relaxation time: tau = eta/E_1
    parameters['nu'] = 0.2
    parameters['stress_state'] = 'plane_strain'

    parameters['time'] = 1.5  # total simulation time in s
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


    # relaxtaion check - first and last value
    eps_r = parameters['u_bc']  # L==1 -> u_bc = eps_r (prescriped strain)
    #
    print(prop2D.p.visco_case)
    if prop2D.p.visco_case.lower() == 'cmaxwell':
        sig0 = parameters['E_0']*eps_r + parameters['E_1']*eps_r
        sigend = parameters['E_0']*eps_r
        print('theory',sig0, sigend)
        print('computed',sig_o_time[0] , sig_o_time[-1])
        assert (sig_o_time[0] - sig0)/sig0 < 1e-8
        assert (sig_o_time[-1]- sigend)/sigend < 1e-4

    # get stresses and strains at the end
    # print('stresses',prop2D.sensors[sensor01.name].data[-1])
    # print('strains',prop2D.sensors[sensor02.name].data[-1])
    strain_xx = prop2D.sensors[sensor02.name].data[-1][0]
    strain_yy = prop2D.sensors[sensor02.name].data[-1][1]
    assert strain_yy == pytest.approx(prop2D.p.u_bc) # L==1!
    assert strain_xx == pytest.approx(-prop2D.p.nu*prop2D.p.u_bc)


    # plot analytic 1D solution against computed (for relaxation test)
    sig_yy = []
    tau = parameters['eta'] / parameters['E_1']
    for i in time:
        sig_yy.append(parameters['E_0'] * eps_r + parameters['E_1'] * eps_r * np.exp(-i / tau))

    # print('analytic 1D == 2D with nu=0', sig_yy)
    # print('stress over time', sig_o_time)

    import matplotlib.pyplot as plt

    plt.plot(time, sig_yy, '*r', label='analytic')
    plt.plot(time, sig_o_time, 'og', label='FEM')
    plt.legend()
    plt.show()


#TODO: add creep test requires load boundary condition not yet in the experiment framework

if __name__ == '__main__':


    test_relaxation_2D()





