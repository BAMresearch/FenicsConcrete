import fenics_concrete
import pytest


def test_e_28_days():
    parameters = fenics_concrete.Parameters()  # using the current default values
    # general
    parameters['log_level'] = 'WARNING'
    # mesh
    parameters['mesh_setting'] = 'left/right'  # default boundary setting
    parameters['dim'] = 2
    parameters['mesh_density'] = 2
    # temperature boundary
    parameters['bc_setting'] = 'full'  # default boundary setting
    parameters['T_0'] = 20  # inital concrete temperature
    parameters['T_bc1'] = 20  # temperature boundary value 1

    parameters['density'] = 2350  # in kg/m^3 density of concrete
    parameters['density_binder'] = 1440  # in kg/m^3 density of the binder
    parameters['themal_cond'] = 2.0  # effective thermal conductivity, approx in Wm^-3K^-1, concrete!
    # self.specific_heat_capacity = 9000  # effective specific heat capacity in J kg⁻1 K⁻1
    parameters['vol_heat_cap'] = 2.4e6  # volumetric heat cap J/(m3 K)
    parameters['b_ratio'] = 0.2  # volume percentage of binder
    parameters['Q_pot'] = 500e3  # potential heat per weight of binder in J/kg
    # p['Q_inf'] = self.Q_pot * self.density_binder * self.b_ratio  # potential heat per concrete volume in J/m3
    parameters['B1'] = 2.916E-4  # in 1/s
    parameters['B2'] = 0.0024229  # -
    parameters['eta'] = 5.554  # something about diffusion
    parameters['alpha_max'] = 0.89  # also possible to approximate based on equation with w/c
    parameters['alpha_tx'] = 0.72  # also possible to approximate based on equation with w/c
    parameters['E_act'] = 5653 * 8.3145  # activation energy in Jmol^-1
    parameters['T_ref'] = 25  # reference temperature in degree celsius
    # setting for temperature adjustment
    parameters['temp_adjust_law'] = 'exponential'
    # polinomial degree
    parameters['degree'] = 2  # default boundary setting
    ### paramters for mechanics problem
    parameters['E'] = 420000  # Youngs Modulus N/m2 or something...
    parameters['nu'] = 0.2  # Poissons Ratio
    # required paramters for alpha to E mapping
    parameters['alpha_t'] = 0.2
    parameters['alpha_0'] = 0.05
    parameters['a_E'] = 0.6
    # required paramters for alpha to tensile and compressive stiffness mapping
    parameters['fc'] = 6210000
    parameters['a_fc'] = 1.2
    parameters['ft'] = 467000
    parameters['a_ft'] = 1.0

    experiment = fenics_concrete.ConcreteCubeExperiment(parameters)
    problem = fenics_concrete.ConcreteThermoMechanical(experiment=experiment, parameters=parameters,
                                                       vmapoutput=False)


    E_sensor = fenics_concrete.sensors.YoungsModulusSensor((0.25, 0.25))
    fc_sensor = fenics_concrete.sensors.CompressiveStrengthSensor((0.25, 0.25))
    doh_sensor = fenics_concrete.sensors.DOHSensor((0.25, 0.25))


    problem.add_sensor(E_sensor)
    problem.add_sensor(fc_sensor)
    problem.add_sensor(doh_sensor)

    # data for time stepping
    dt = 3600  # 60 min step
    time = dt * 1000  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    # initialize time
    t = dt  # first time step time

    doh = 0
    while doh < parameters['alpha_tx']:  # time


        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this

        # prepare next timestep
        t += dt
        print("t: ", t)
        print("doh: ", problem.sensors[doh_sensor.name].data[-1])
        print("  E: ", problem.sensors[E_sensor.name].data[-1])

    # get last measure value
        doh = problem.sensors[doh_sensor.name].data[-1]

    assert problem.sensors[E_sensor.name].data[-1] == pytest.approx(parameters['E'], 0.1)

    #assert problem.sensors[fc_sensor.name].data[-1] == pytest.approx(parameters['fc_inf'], 0.1)
