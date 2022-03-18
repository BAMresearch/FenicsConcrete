import concrete_model

import pytest

# import warnings
# from ffc.quadrature.deprecation \
#     import QuadratureRepresentationDeprecationWarning
# warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

def simple_simulation(sensor):

    parameters = concrete_model.Parameters()  # using the current default values
    # general
    parameters['log_level'] = 'WARNING'
    # mesh
    parameters['mesh_setting'] = 'left/right'  # default boundary setting
    parameters['dim'] = 2
    parameters['mesh_density'] = 4
    # temperature boundary
    parameters['bc_setting'] = 'test-setup'  # default boundary setting
    parameters['T_0'] = 10  # inital concrete temperature
    parameters['T_bc1'] = 20  # temperature boundary value 1
    parameters['T_bc2'] = 30  # temperature boundary value 2

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
    parameters['alpha_max'] = 0.875  # also possible to approximate based on equation with w/c
    parameters['E_act'] = 5653 * 8.3145  # activation energy in Jmol^-1
    parameters['T_ref'] = 25  # reference temperature in degree celsius
    # setting for temperature adjustment
    parameters['temp_adjust_law'] = 'exponential'
    # polinomial degree
    parameters['degree'] = 2  # default boundary setting
    ### paramters for mechanics problem
    parameters['E_28'] = 15000000  # Youngs Modulus N/m2 or something...
    parameters['nu'] = 0.2  # Poissons Ratio
    # required paramters for alpha to E mapping
    parameters['alpha_t'] = 0.2
    parameters['alpha_0'] = 0.05
    parameters['a_E'] = 0.6
    # required paramters for alpha to tensile and compressive stiffness mapping
    parameters['fc_inf'] = 6210000
    parameters['a_fc'] = 1.2
    parameters['ft_inf'] = 467000
    parameters['a_ft'] = 1.0

    experiment = concrete_model.get_experiment('ConcreteCube', parameters)
    problem = concrete_model.ConcreteThermoMechanical(experiment, parameters)

    problem.add_sensor(sensor)

    # data for time stepping
    dt = 3600  # 60 min step
    time = dt * 10  # total simulation time in s

    # set time step
    problem.set_timestep(dt)  # for time integration scheme

    # initialize time
    t = dt  # first time step time


    while t <= time:  # time
        # solve temp-hydration-mechanics
        problem.solve(t=t)  # solving this

        # prepare next timestep
        t += dt

    # get last measure value
    data = problem.sensors[0].data[-1][1]

    return data



def test_temperature():

    sensor = concrete_model.sensors.TemperatureSensor((0.25, 0.25))

    data = simple_simulation(sensor)
    
    assert float(data) == pytest.approx(23.847730968641713)


def test_max_temperature():

    sensor = concrete_model.sensors.MaxTemperatureSensor()

    data = simple_simulation(sensor)
    
    assert float(data) == pytest.approx(31.90459255375174)


def test_degree_of_hydration():

    sensor = concrete_model.sensors.DOHSensor((0.25, 0.25))

    data = simple_simulation(sensor)
    
    assert float(data) == pytest.approx(0.16581303886083476)


def test_displacement():

    sensor = concrete_model.sensors.DisplacementSensor((0.25, 0.25))

    data = simple_simulation(sensor)
    
    assert float(data) == pytest.approx(-0.0002136038620005609)
