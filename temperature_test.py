from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_model as model

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------
# Create mesh and define function space
nx = ny = 20
mesh = UnitSquareMesh(nx, ny)

# problem setup
mat = model.ConcreteMaterialData() # setting up some basic material things
concrete_problem = model.ConcreteTempHydrationModel(mesh, mat, pv_name='pv_output')  #setting up the material problem, with material data and mesh

# Define boundary and initial conditions
T_boundary = 30+273.15  # input in celcius???
T_zero = 20+273.15 # input in celcius???

# Initial temp. condition
concrete_problem.set_initialT(T_zero)

# Temperature boundary conditions
def full_boundary(x, on_boundary):
    return on_boundary
def empty_boundary(x, on_boundary):
    return None

T_bc = Expression('t_boundary', t_boundary = T_boundary, degree= 0)
#bc = DirichletBC(concrete_problem.V, T_bc, empty_boundary)
bc = DirichletBC(concrete_problem.V, T_bc, full_boundary)

concrete_problem.set_bcs(bc)

# data for time stepping
#time steps
dt = 60*20 # time step
hours = 48
time = hours*60*60         # total simulation time in s
# set time step
concrete_problem.set_timestep(dt) # for time integration scheme

#initialize time
t = 0 # first time step time

# plot data fields
plot_data = [[],[],[],[],[],[],[],[],[]]

# setting up the solver
solver = NewtonSolver()

while t <= time:
    print('time =', t)
    print('Solving: T')
    solver.solve(concrete_problem, concrete_problem.T.vector())

    # prepare next timestep
    t += dt
    concrete_problem.update_history()

    # Plot temperature and degree of hydration
    concrete_problem.pv_plot(t=t)

    #plot points
    plot_data[0].append(t)
    plot_data[1].append(concrete_problem.T(0.0,0.0)-273.15)
    plot_data[2].append(concrete_problem.T(0.25,0.25)-273.15)
    plot_data[3].append(concrete_problem.T(0.5,0.5)-273.15)
    plot_data[4].append(concrete_problem.q_alpha.vector().get_local()[0])


#plotting stuff
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
ax1.plot(plot_data[0], plot_data[1])
ax1.plot(plot_data[0], plot_data[2])
ax1.plot(plot_data[0], plot_data[3])
ax1.set_title('Temperature')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Temperature')
ax2.plot(plot_data[0], plot_data[4]) #alpha

fig.suptitle('Some points', fontsize=16)

plt.show()

# some copied functions from fenics constitutive

