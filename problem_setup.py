from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_model as model

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------
# Create mesh and define function space
dim = 2
n = 20
if dim == 2:
    mesh = UnitSquareMesh(n, n)
elif dim == 3:
    mesh = UnitCubeMesh(n, n, n)
else:
    print(f'wrong dimension {dim}')
    exit()


# problem setup
mat = model.ConcreteMaterialData() # setting up some basic material things
mat.set_parameters('CostActionTeam2')

# setting up new strucutre
concrete_problem = model.ConcreteModel(mesh, mat)

# Define boundary and initial conditions
T_boundary = 10+273.15  # input in celcius???
T_zero = 20 # input in celcius, computation in kelvin
T_L = 50+273.15 # input in celcius???
T_R = 10+273.15 # input in celcius???

# Initial temp. condition
concrete_problem.set_inital_T(T_zero) #TODO add check before solve to check for manual things..., do a log file...???

# Temperature boundary conditions
def full_boundary(x, on_boundary):
    return on_boundary
def L_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0)
def LU_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0) or on_boundary and near(x[1], 0)
def R_boundary(x, on_boundary):
    return on_boundary and near(x[0], 1)
def empty_boundary(x, on_boundary):
    return None

T_bc = Expression('t_boundary', t_boundary = T_boundary, degree= 0)
T_bcL = Expression('t_boundary', t_boundary = T_L, degree= 0)
T_bcR = Expression('t_boundary', t_boundary = T_R, degree= 0)
#bc = DirichletBC(concrete_problem.V, T_bc, empty_boundary)

bc = []
#bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
bc.append(DirichletBC(concrete_problem.temperature_problem.V, T_bcR, LU_boundary))
bc.append(DirichletBC(concrete_problem.temperature_problem.V, T_bcL, R_boundary))
concrete_problem.set_temperature_bcs(bc)
# stuff for mechanics problem

# displacement boundary conditions
# define surfaces
def bottom_surface(x, on_boundary):
    return on_boundary and near(x[1], 0)

bcm = DirichletBC(concrete_problem.mechanics_problem.V, Constant((0,0)), bottom_surface)
concrete_problem.set_displacement_bcs(bcm)

# data for time stepping
#time steps
dt = 60*20 # time step
hours = 50
time = hours*60*60         # total simulation time in s
# set time step
concrete_problem.set_timestep(dt) # for time integration scheme
#

#initialize time
t = dt # first time step time

# plot data fields


plot_data = [[],[],[],[],[],[],[],[],[]]


while t <= time:

    print('time =', t)

    # solve temp-hydration-mechanics
    concrete_problem.solve() # solving this

    # plot fields
    concrete_problem.pv_plot(t=t)
    # prepare next timestep
    t += dt

    # #plot points
    # plot_data[0].append(t)
    # plot_data[1].append(temperature_problem.T(0.0, 0.0) - 273.15)
    # plot_data[2].append(temperature_problem.T(0.25, 0.25) - 273.15)
    # plot_data[3].append(temperature_problem.T(0.5, 0.5) - 273.15)
    # plot_data[4].append(temperature_problem.q_alpha.vector().get_local()[0])


# #plotting stuff
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
# ax1.plot(plot_data[0], plot_data[1])
# ax1.plot(plot_data[0], plot_data[2])
# ax1.plot(plot_data[0], plot_data[3])
# ax1.set_title('Temperature')
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Temperature')
# ax2.plot(plot_data[0], plot_data[4]) #alpha
#
# fig.suptitle('Some points', fontsize=16)
#
# plt.show()

# some copied functions from fenics constitutive

