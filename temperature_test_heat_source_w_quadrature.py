"""
FEniCS tutorial demo program: Diffusion equation with Dirichlet
conditions and a solution that will be exact at all nodes on
a uniform mesh.
Difference from heat.py: A (coeff.matrix) is assembled
only once.
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Create mesh and define function space
nx = ny = 4
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary and initial conditions
t_boundary = 25+273.15  # input in celcius
t_zero = 20+273.15 # input in celcius
alpha_zero = 0.5
t0 = Expression('t_zero', t_zero=t_zero, degree= 0)
alpha0 = Expression('alpha0', alpha0=alpha_zero, degree= 0)
u0 = Expression('t_boundary', t_boundary = t_boundary, degree= 1)

def full_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, full_boundary)
bc_alpha = DirichletBC(V, Constant(0.01), full_boundary) # why does this work??? not full_boundary(..something...)

# Initial temp. condition
T_n = interpolate(t0, V)
# Initial hydration condition
alpha_n = interpolate(alpha0, V)

#time steps
dt = 60*60# time step
hours = 48
time = hours*60*60          # total simulation time in s

# Material parameter
themal_cond_eff = 1.6 # effective thermal conductivity, approx in Wm^-1K^-1
density = 2350 # in kg/m^3
specific_heat_capacity = 900 # effective specific heat capacity in J kg⁻1 K⁻1
vol_heat_cap = density*specific_heat_capacity
# values from book for CEM I 52.5
Q_inf = 505900 # potential heat approx. in J/kg?? ... TODO : maybe change units to kg??? or mutiply with some binder value...
B1 = Constant(3.79E-4)# in 1/s
B2 = Constant(6E-5) # -
eta = Constant(5.8) # something about diffusion
alpha_max_scalar = 0.85 # also possible to approximate based on equation with w/c
alpha_max_expr = Expression('alpha_max', alpha_max=alpha_max_scalar, degree= 0)
alpha_max = interpolate(alpha_max_expr, V)
activation_energy = 38300 # in Jmol^-1
T_ref = 25 # degree celsius

# start with "real" fenics problem
# Define variational problem
T = Function(V) #temperature
alpha = Function(V) #alpha, degree of hydration
vT = TestFunction(V)
valpha = TestFunction(V)

dummy = 1800 # some test constant

#quadrature stuff for x*T

q = "Quadrature"
cell = mesh.ufl_cell()
q_dim = 1 # only temperature...
deg_q = 1 # I think...

QV = VectorElement(q, cell, deg_q, quad_scheme="default", dim=q_dim)
QT = TensorElement(q, cell, deg_q, quad_scheme="default", shape=(q_dim, q_dim))
VQV, VQT = [FunctionSpace(mesh, Q) for Q in [QV, QT]]

q_source = Function(VQV, name="heat source")
q_eps = Function(VQV, name="strains")
q_dsource_dtemp = Function(VQT, name="source_temperature tangent")

metadata = {"quadrature_degree": deg_q, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

# How do I define these... ???? ...???
# R = -inner(eps(u_), q_sigma) * dxm
# dR = inner(eps(du), dot(q_dsigma_deps, eps(u_))) * dxm


# heat problem
# test problem with heatsource as x*temperature for testing purposes
F_T = vol_heat_cap*(T)*vT*dxm + dt*dot(themal_cond_eff*grad(T), grad(vT))*dxm - vol_heat_cap*T_n*vT*dxm - dummy*T*vT*dxm

a = vol_heat_cap*(T)*vT*dxm + dt*dot(themal_cond_eff*grad(T), grad(vT))*dxm - dummy*T*vT*dxm
L = vol_heat_cap*T_n*vT*dxm
#a = lhs(F_T)
#L = rhs(F_T)

#a = vol_heat_cap*(T)*vT*dx + dt*dot(themal_cond_eff*grad(T), grad(vT))*dx - dummy*T*vT*dx
#L = vol_heat_cap*T_n*vT*dx



t = dt

# visualize initial conditions in paraview
pv_file = XDMFFile('output_paraview.xdmf')
pv_file.parameters["flush_output"] = True
pv_file.parameters["functions_share_mesh"] = True
# Plot fields
temperature = Function(V, name='Temperature')
doh = Function(V, name='Degree of hydration')

# paraview export
temperature.assign(T_n)
pv_file.write(temperature,0) # Save solution to file in XDMF format
doh.assign(alpha_n)
pv_file.write(doh,0) # Save solution to file in XDMF format

# plot data fields
plot_data = [[[],[],[]],[[],[],[]],[[],[],[]]]

while t <= time:
    print('time =', t)

    print('Solving: T')
    # solve temperature

    #A = assemble(a)
    #b = assemble(L)
    #bc.apply(A,b)

    #solve(A,T.vector(),b)



    solve(a - L == 0, T, bc)
    #solve(F_T == 0, T, bc)

    # prepare next timestep
    t += dt
    T_n.assign(T)
    alpha_n.assign(alpha)


    # Output Data
    # Plot displacements
    temperature.assign(T)
    pv_file.write(temperature,t) # Save solution to file in XDMF format
    doh.assign(alpha)
    pv_file.write(doh,t) # Save solution to file in XDMF format

    #plot points
    plot_data[0][0].append(t)
    plot_data[0][1].append(temperature(0.0,0.0)-273.15)
    plot_data[0][2].append(doh(0.,0.))

    plot_data[1][0].append(t)
    plot_data[1][1].append(temperature(0.25,0.25)-273.15)
    plot_data[1][2].append(doh(0.2,0.2))

    plot_data[2][0].append(t)
    plot_data[2][1].append(temperature(0.5,0.5)-273.15)
    plot_data[2][2].append(doh(0.5,0.5))


#plotting stuff
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
ax1.plot(plot_data[0][0], plot_data[0][1])
ax1.plot(plot_data[1][0], plot_data[1][1])
ax1.plot(plot_data[2][0], plot_data[2][1])
ax1.set_title('Temperature')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Temperature')

ax2.plot(plot_data[0][0], plot_data[0][2])
ax2.plot(plot_data[1][0], plot_data[1][2])
ax2.plot(plot_data[2][0], plot_data[2][2])
ax2.set_xlabel('time (s)')
ax2.set_title('Degree of hydration')

fig.suptitle('Some points', fontsize=16)

plt.show()
