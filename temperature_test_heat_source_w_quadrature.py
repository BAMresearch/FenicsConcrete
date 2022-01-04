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
import scipy.optimize
import time as timer

# helper functions...
def set_q(q, values):
    """
    q:
        quadrature function space
    values:
        entries for `q`
    """
    v = q.vector()
    v.zero()
    v.add_local(values.flat)
    v.apply("insert")

class LocalProjector:
    def __init__(self, expr, V, dxm):
        """
        expr:
            expression to project
        V:
            quadrature function space
        dxm:
            dolfin.Measure("dx") that matches V
        """
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_) * dxm
        b_proj = inner(expr, v_) * dxm
        self.solver = LocalSolver(a_proj, b_proj)
        self.solver.factorize()

    def __call__(self, u):
        """
        u:
            function that is filled with the solution of the projection
        """
        self.solver.solve_local_rhs(u)

# stuff relevant to compute the degree of hydration at the integration points...
class ConcreteMaterialData:
    def __init__(self):
        self.zeroC = 273.15 # temperature in kelvon for zero degree celsius
        # Material parameter for concrete model with temperature and hydration
        self.density = Constant(2350)  # in kg/m^3
        self.themal_cond = Constant(1.6)  # effective thermal conductivity, approx in Wm^-1K^-1
        self.specific_heat_capacity = Constant(900)  # effective specific heat capacity in J kg⁻1 K⁻1
        self.vol_heat_cap = self.density * self.specific_heat_capacity
        # values from book for CEM I 52.5
        self.Q_inf = Constant(5059000)  # potential heat approx. in J/kg?? ... TODO : maybe change units to kg??? or mutiply with some binder value...
        self.B1 = 3.79E-4  # in 1/s
        self.B2 = 6E-5  # -
        self.eta = 5.8  # something about diffusion
        self.alpha_max = 0.85  # also possible to approximate based on equation with w/c
        self.E_act = 38300  # activation energy in Jmol^-1
        self.igc = 8.3145  # ideal gas constant [JK −1 mol −1 ]
        self.T_ref_celsius = 25  # reference temperature in degree celsius #TODO figure out how/when where to wirk with celcisus and there with kelvin
        self.T_ref = self.T_ref_celsius + self.zeroC # reference temperature in degree celsius #TODO figure out how/when where to wirk with celcisus and there with kelvin
        # setting for temperature adjustment
        # option: 'exponential' and 'off'
        self.temp_adjust_law = 'exponential'

    #TODO find out if this forumla deals with kelvin or celsius???

    # TODO how do I implement affinity function and temp adjust such that I can change them in material def but not in problem setting???

    # temperature adjustmeent factor
    def temp_adjust(self, T):
        val = 1
        if self.temp_adjust_law == 'exponential':
            val = np.exp(-self.E_act/self.igc*(1/T-1/self.T_ref))
        elif self.temp_adjust_law == 'off':
            pass
        else:
            #TODO throw correct error
            print(f'Warning: Incorrect temp_adjust_law {self.temp_adjust_law} given')
            print('*******  Only exponential and off implemented')
        return val

    # derivative of the temperature adjustment factor with respect to the temperature
    def temp_adjust_tangent(self, T):
        val = 0
        if self.temp_adjust_law == 'exponential':
            val = self.E_act/self.igc/T**2
        return val

    # affinity function
    def affinity(self, delta_alpha, alpha_n):
        affinity = self.B1 * (self.B2 / self.alpha_max + delta_alpha + alpha_n) * (self.alpha_max - (delta_alpha + alpha_n)) * np.exp(-self.eta * (delta_alpha + alpha_n) / self.alpha_max)
        return affinity

    # derivative of affinity with respect to delta alpha
    def daffinity_ddalpha(self, delta_alpha, alpha_n):
        affinity_prime = self.B1 * np.exp(-self.eta * (delta_alpha + alpha_n) / self.alpha_max) * ((self.alpha_max - (delta_alpha + alpha_n)) * (self.B2 / self.alpha_max + (delta_alpha + alpha_n)) * ( -self.eta / self.alpha_max) - self.B2 / self.alpha_max - 2 * (delta_alpha + alpha_n) + self.alpha_max)
        return affinity_prime

class ConcreteTempHydrationModel(NonlinearProblem):
    def __init__(self, mesh, mat, **kwargs):
        NonlinearProblem.__init__(self) # apparently required to initialize things
        self.mat = mat # object with material data, parameters etc...
        deg_q = 1  # quadrature degree, TODO, should/could this be based on the mesh?

        metadata = {"quadrature_degree": deg_q, "quadrature_scheme": "default"}
        dxm = dx(metadata=metadata)

        # solution field
        self.V = FunctionSpace(mesh, 'P', 1)


        # generic quadrature function space
        cell = mesh.ufl_cell()
        q = "Quadrature"
        quadrature_element = FiniteElement(q, cell, degree=deg_q, quad_scheme="default")
        q_V = FunctionSpace(mesh, quadrature_element)

        # quadrature functions
        self.q_dummy = Function(q_V, name="dummy field")
        self.q_T = Function(q_V, name="temperature")
        self.q_alpha = Function(q_V, name="degree of hydration")
        #how to initialize alpha_n???? do I need to? shouldnt it always be zero???
        self.q_alpha_n = Function(q_V, name="degree of hydration last time step")
        self.q_delta_alpha = Function(q_V, name="degree of hydration last time step")
        #self.q_A = Function(q_V, name="affinity")
        #self.q_g = Function(q_V, name="temperature correction")

        self.q_dalpha_dT = Function(q_V, name="current degree of hydration")
        #self.q_dg_dT = Function(q_V, name="temperature correction tangent")

        # Define variational problem
        self.T = Function(self.V)  # temperature
        self.T_n = Function(self.V)  # overwritten later...
        T_ = TrialFunction(self.V) # temperature
        #alpha = Function(self.V)  # alpha, degree of hydration
        vT = TestFunction(self.V)
        #valpha = TestFunction(self.V)

        # inital value TODO: check if it is reset
        # TODO do I need it to be a Constant???
        self.dt = Constant(0)       # TODO somehow make sure this is reset!


        # test problem with heatsource as x*temperature for testing purposes
        # normal form
        R_ufl = mat.vol_heat_cap * (self.T) * vT * dxm
        R_ufl += self.dt * dot(mat.themal_cond * grad(self.T),grad(vT)) * dxm
        R_ufl += - mat.vol_heat_cap * self.T_n * vT * dxm
        # quadrature point part
        self.R = R_ufl - mat.Q_inf * self.q_delta_alpha * vT * dxm

        # derivative
        # normal form
        dR_ufl = derivative(R_ufl, self.T)
        # quadrature part
        self.dR = dR_ufl - mat.Q_inf * self.q_dalpha_dT * T_ * vT * dxm

        # setup projector to project continuous funtionspace to quadrature
        self.project_T = LocalProjector(self.T, q_V, dxm)

        self.assembler = None  #set as default, to check if bc have been added???


    def delta_alpha_fkt(self,delta_alpha, alpha_n, T):
        return delta_alpha - float(self.dt) * self.mat.affinity(delta_alpha, alpha_n) * self.mat.temp_adjust(T)


    def delta_alpha_prime(self,delta_alpha, alpha_n, T):
        return 1 - float(self.dt) * self.mat.daffinity_ddalpha(delta_alpha, alpha_n) * self.mat.temp_adjust(T)


    def evaluate_material(self):
        # project stuff (temperautre) onto their quadrature spaces
        self.project_T(self.q_T)
        temperature_list = self.q_T.vector().get_local()

        # get alpha values
        alpha_n_list = self.q_alpha_n.vector().get_local()





        guess = np.full(np.shape(alpha_n_list), 0.2)
        # I need a numpy list for temp. alpha_n and the guess!!!
        delta_alpha_list = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha_n_list, temperature_list),fprime=self.delta_alpha_prime, x0=guess)
        alpha_list = alpha_n_list + delta_alpha_list
        dalpha_dT_list = float(self.dt) * self.mat.affinity(alpha_list, alpha_n_list)* self.mat.temp_adjust_tangent(temperature_list)







        set_q(self.q_alpha, alpha_list)
        set_q(self.q_delta_alpha, delta_alpha_list)
        set_q(self.q_dalpha_dT, dalpha_dT_list)


    def update(self):
        # TODO ???
        # when or what do I need to update???
        # self.calculate_e(self.q_e)
        # self.mat.update(self.q_e.vector().get_local())
        # set_q(self.q_k, self.mat.kappa)  # just for post processing
        pass

    def update_history(self):
        self.T_n.assign(self.T) # save temparature field
        self.q_alpha_n.assign(self.q_alpha) # save alpha field

    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        # if self.dt == Constant(0):
        #     raise RuntimeError("You need to define a time step `.dt` other than zero before the solve!")
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)

#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------
# Create mesh and define function space
nx = ny = 20
mesh = UnitSquareMesh(nx, ny)

# problem setup
mat = ConcreteMaterialData() # setting up some basic material things
# todo, maybe add linear/quadratic etc... as a paramter???
concrete_problem = ConcreteTempHydrationModel(mesh, mat)  #setting up the material problem, with material data and mesh

# Define boundary and initial conditions
t_boundary = 30+273.15  # input in celcius???
t_zero = 30+273.15 # input in celcius???

# TODO should initial condition be treated "inside" and controlled as "material" paramter????

t0 = Expression('t_zero', t_zero=t_zero, degree= 0)
u0 = Expression('t_boundary', t_boundary = t_boundary, degree= 1)

def full_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(concrete_problem.V, u0, full_boundary)

concrete_problem.set_bcs(bc)

# TODO is this correctly implemented, is there a better/easier way????
# Initial temp. condition
concrete_problem.T_n.interpolate(t0)
# might be necessary for first interation step... TODO: what about boundary values???
concrete_problem.T.interpolate(t0)

# data for time stepping
#time steps
dt = 60*20# time step
hours = 48
time = hours*60*60         # total simulation time in s

# set them directly in the class
# todo maybe rather add a function ".set_timestep(dt)" ???
concrete_problem.dt.assign(Constant(dt)) # for time integration scheme

t = dt # first time step time

# visualize initial conditions in paraview
pv_file = XDMFFile('output_paraview.xdmf')
pv_file.parameters["flush_output"] = True
pv_file.parameters["functions_share_mesh"] = True
# Plot fields
temperature = Function(concrete_problem.V, name='Temperature')
#alpha = Function(concrete_problem.V, name='alpha')

# paraview export
temperature.assign(concrete_problem.T_n)
pv_file.write(temperature,0) # Save solution to file in XDMF format

# plot data fields
plot_data = [[],[],[],[],[],[],[],[],[]]


solver = NewtonSolver()


#time
#while t <= time:
while t <= time:
    print('time =', t)
    print('Solving: T')
    solver.solve(concrete_problem, concrete_problem.T.vector())

    # prepare next timestep
    t += dt
    concrete_problem.update_history()

    # Output Data
    # Plot displacements
    temperature.assign(concrete_problem.T)
    #alpha.assign(concrete_problem.q_alpha)
    pv_file.write(temperature,t) # Save solution to file in XDMF format
    #plot points
    plot_data[0].append(t)
    plot_data[1].append(temperature(0.0,0.0)-273.15)
    plot_data[2].append(temperature(0.25,0.25)-273.15)
    plot_data[3].append(temperature(0.5,0.5)-273.15)
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

