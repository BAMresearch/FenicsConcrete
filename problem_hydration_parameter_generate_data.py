from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------

# initiate material problem
material_problem = concrete_problem.ConcreteThermoMechanical()
# get the respective function
hydration_fkt = material_problem.get_heat_of_hydration_ftk()

# set required parameter
parameter = concrete_experiment.Parameters() # using the current default values

parameter['B1'] = 2.916E-4  # in 1/s
parameter['B2'] = 0.0024229  # -
parameter['eta'] = 5.554  # something about diffusion
parameter['alpha_max'] = 0.875  # also possible to approximate based on equation with w/c
parameter['E_act'] = 47002   # activation energy in Jmol^-1
parameter['T_ref'] = 25  # reference temperature in degree celsius
parameter['igc'] = 8.3145  # ideal gas constant in [J/K/mol], CONSTANT!!!
parameter['zero_C'] = 273.15  # in Kelvin, CONSTANT!!!
parameter['Q_pot'] = 500e3 # potential heat per weight of binder in J/kg

# additional function values
time = 60*60*24*28
dt = 60*30
T = 25

for val in  [15,20,25,30,35,40,45,50,60,70,80]:
    T = val

    time_list, heat_list, doh_list = hydration_fkt(T,time,dt,parameter)
    #print(output)
    f = open(f"test_hydration_data_T{T}.dat", "w")
    f.write(f'# Artificial heat of hydration data, at temperature {T}\n')
    f.write(f'# With material paramters:\n')
    f.write(f'# B1: {parameter.B1}\n')
    f.write(f'# B2: = {parameter.B2}\n')
    f.write(f'# eta: = {parameter.eta}\n')
    f.write(f'# alpha_max: = {parameter.alpha_max}\n')
    f.write(f'# E_act: = {parameter.E_act}\n')
    f.write(f'# T_ref: = {parameter.T_ref}\n')
    f.write(f'# Q_pot: = {parameter.Q_pot}\n')
    f.write(f'# and numerical parameters\n')
    f.write(f'# dt: = {dt}\n')
    f.write(f'# total time: = {time}\n')
    f.write(f'#\n')
    word1, word2 = 'Time in hours', 'Heat released'
    f.write(f'# {word1:20} {word2:20}\n')
    for i in range(len(time_list)):
        f.write(f'  {time_list[i]:20.8f} {heat_list[i]:20.8f}\n')
    f.close()

    plt.plot(time_list,heat_list)

plt.show()
