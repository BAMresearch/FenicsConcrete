from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------


# 1: read the experimental data

# read data
time_data = []
heat_data = []

T_datasets = []

# extract data from csv file
with open('cost_action_hydration_data.csv') as f:
    for i,line in enumerate(f):
        if i == 0:
            split_line = line.split(',')
            for j in range(0,len(split_line),2):
                degree = split_line[j].split('_')[0]
                T_datasets.append(float(degree.strip()))
                time_data.append([])
                heat_data.append([])
        if i > 1:
            split_line = line.split(',')
            for j in range(len(T_datasets)):
                print(i,j,split_line[j*2],split_line[j*2+1])
                if split_line[j*2].strip() != '':
                    time_data[j].append(float(split_line[j*2].strip())*60*60) # convert to seconds
                    heat_data[j].append(float(split_line[j*2+1].strip()))


# sort data!!!
for i in range(len(heat_data)):
    zipped_lists = zip(time_data[i], heat_data[i])
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    time_data[i], heat_data[i] = [ list(tuple) for tuple in  tuples]


# 2: define parameter sets
parameter = concrete_experiment.Parameters() # using the current default values
# general values
parameter['alpha_max'] = 0.875  # also possible to approximate based on equation with w/c
parameter['T_ref'] = 20  # reference temperature in degree celsius
parameter['igc'] = 8.3145  # ideal gas constant in [J/K/mol], CONSTANT!!!
parameter['zero_C'] = 273.15  # in Kelvin, CONSTANT!!!
parameter['Q_pot'] = 500e3 # potential heat per weight of binder in J/kg

parameter_sets = []

parameter_paper = parameter
parameter_paper['B1'] = 2.916e-4  # in 1/s
parameter_paper['B2'] = 0.0024229  # -
parameter_paper['eta'] = 5.554  # something about diffusion
parameter_paper['E_act'] = 47002   # activation energy in Jmol^-1
parameter_sets.append(parameter_paper)

parameter_fav = parameter
parameter_fav['B1'] = 0.000255  # in 1/s
parameter_fav['B2'] = 0.000477  # -
parameter_fav['eta'] = 5.597  # something about diffusion
parameter_fav['E_act'] = 36450   # activation energy in Jmol^-1
parameter_sets.append(parameter_fav)

parameter_simple = parameter
parameter_simple['B1'] = 0.00045  # in 1/s
parameter_simple['B2'] = 3.82e-5  # -
parameter_simple['eta'] = 6.1355  # something about diffusion
parameter_simple['E_act'] = 32144   # activation energy in Jmol^-1
parameter_sets.append(parameter_simple)

parameter_start = parameter
parameter_start['B1'] = 0.00033  # in 1/s
parameter_start['B2'] = 0.00038  # -
parameter_start['eta'] = 5.12  # something about diffusion
parameter_start['E_act'] = 32511   # activation energy in Jmol^-1
parameter_sets.append(parameter_start)

# 3: plot each set and experimental data

# initiate material problem
material_problem = concrete_problem.ConcreteThermoMechanical()
# get the respective function
hydration_fkt = material_problem.get_heat_of_hydration_ftk()

# generate a time list for plotting
# get max time
tmax = 0
for i in range(len(time_data)):
    if time_data[i][-1] > tmax:
        tmax = time_data[i][-1]

dt = 300
plot_time_list = np.arange(0, tmax, dt)

# genrate data from model
for parameter in  parameter_sets:
    for i,T in enumerate(T_datasets):

    heat_list = hydration_fkt(T,plot_time_list,dt,parameter)





    plt.plot(time_list,heat_list)

plt.show()
