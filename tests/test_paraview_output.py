import numpy as np

import fenics_concrete

import pytest

import xml.etree.ElementTree as ET

import os

def simple_simulation(new_parameters, name):

    parameters = fenics_concrete.Parameters()  # using the current default values
    # general
    parameters['log_level'] = 'WARNING'
    # mesh
    parameters['mesh_setting'] = 'left/right'  # default boundary setting
    parameters['bc_setting'] = 'test-setup'  # default boundary setting
    parameters['mesh_density'] = 4
    # temperature boundary
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
    parameters['E_28'] = 15000000  # Youngs Modulus N/m2 or something... TODO: check units!
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

    parameters = parameters + new_parameters

    experiment = fenics_concrete.ConcreteCubeExperiment(parameters)

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    problem = fenics_concrete.ConcreteThermoMechanical(experiment, parameters, pv_name=file_path+'test_'+name)


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
        problem.pv_plot(t=t)

        # prepare next timestep
        t += dt



def compare_pv_files(ref_file, test_file):
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    
    #   better compare the files...
    root_ref = ET.parse(file_path+ref_file).getroot()
    test_ref = ET.parse(file_path+test_file).getroot()

    # loop over all timesteps
    for ref_step, test_step in zip(root_ref[0][0], test_ref[0][0]):
        # checking general information
        for ref_element, test_element in zip(ref_step, test_step):
            assert ref_element.tag == test_element.tag
            assert ref_element.attrib == test_element.attrib

        ref_data = ref_step[3:]
        test_data = test_step[3:]
        # checking the saved data itself
        for ref_dataset, test_dataset in zip(ref_data, test_data):
            ref_list = np.array(list(map(float, ref_dataset[0].text.split())))
            test_list = np.array(list(map(float, ref_dataset[0].text.split())))
            assert ref_list ==  pytest.approx(test_list)



@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1,2])
def test_pv_output(dim,degree):
    file_name = str(dim) + 'D_degr' + str(degree)
    parameters = fenics_concrete.Parameters()  # using the current default values
    parameters['dim'] = dim
    parameters['degree'] = degree  # default boundary setting

    simple_simulation(parameters, file_name)

    compare_pv_files('ref_'+file_name+'.xdmf','test_'+file_name+'.xdmf')
    
