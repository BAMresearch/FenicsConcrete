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
class concrete_material_data:
    def __init__(self):
        self.zeroC = 273.15 # temperature in kelvon for zero degree celsius
        # Material parameter for concrete model with temperature and hydration
        self.density = Constant(2350)  # in kg/m^3
        self.themal_cond = Constant(1.6)  # effective thermal conductivity, approx in Wm^-1K^-1
        self.specific_heat_capacity = Constant(900)  # effective specific heat capacity in J kg⁻1 K⁻1
        self.vol_heat_cap = self.density * self.specific_heat_capacity
        # values from book for CEM I 52.5
        #self.Q_inf = Constant(505900)  # potential heat approx. in J/kg?? ... TODO : maybe change units to kg??? or mutiply with some binder value...
        #self.B1 = Constant(3.79E-4)  # in 1/s
        #self.B2 = Constant(6E-5)  # -
        #self.eta = Constant(5.8)  # something about diffusion
        #self.alpha_max = Constant(0.85)  # also possible to approximate based on equation with w/c
        #self.E_act = 38300  # activation energy in Jmol^-1
        #self.T_ref_celsius = 25  # reference temperature in degree celsius #TODO figure out how/when where to wirk with celcisus and there with kelvin
        #self.T_ref = self.T_ref_celsius + self.zeroC # reference temperature in degree celsius #TODO figure out how/when where to wirk with celcisus and there with kelvin

        # Young's modulus          [N/mm²]
        #self.E = 20000.0
        # damage law
        #self.dmg = damage_exponential
        # define some material paramers... maybe depending on concrete type??


    #
    #
    # def kappa_kkt(self, e):
    #     if self.kappa is None:
    #         self.kappa = self.ft / self.E
    #     return np.maximum(e, self.kappa)
    #
    # def integrate(self, eps_flat, e):
    #     kappa = self.kappa_kkt(e)
    #     dkappa_de = (e >= kappa).astype(int)
    #
    #     eps = eps_flat.reshape(-1, 3)
    #     self.sigma, self.dsigma_deps, self.dsigma_de = hooke(
    #         self, eps, kappa, dkappa_de
    #     )
    #     self.eeq, self.deeq = modified_mises_strain_norm(self, eps_flat)
    #
    # def update(self, e):
    #     self.kappa = self.kappa_kkt(e)

class concrete_temp_hydration_model(NonlinearProblem):
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
        #self.q_T = Function(q_V, name="temperature")
        #self.q_alpha = Function(q_V, name="degree of hydration")
        #self.q_A = Function(q_V, name="affinity")
        #self.q_g = Function(q_V, name="temperature correction")

        #self.q_dalpha_dT = Function(q_V, name="current degree of hydration")
        #self.q_dg_dT = Function(q_V, name="temperature correction tangent")

        # Define variational problem
        self.T = Function(self.V)  # temperature
        self.T_n = Function(self.V)  # overwritten later...
        #T_ = TrialFunction(self.V) # temperature
        #alpha = Function(self.V)  # alpha, degree of hydration
        vT = TestFunction(self.V)
        #valpha = TestFunction(self.V)

        # setup form and derivative
        # material parameters from mat
        # thermal con eff
        # vol heat cap
        # more????
        self.dt = Constant(0)       # TODO somehow make sure this is reset!



        # try to fill q_dummy with different values at each point
        # n_gauss = len(self.q_dummy.vector())
        # dummy_list = np.zeros(n_gauss)
        # # applying different values at each quadrature point, as a test
        # for i in range(n_gauss):
        #     dummy_list[i] = i/n_gauss*3600 # 1800
        # set_q(self.q_dummy, dummy_list)

        # test problem with heatsource as x*temperature for testing purposes
        # normal form
        R_ufl = mat.vol_heat_cap * (self.T) * vT * dxm
        R_ufl += self.dt * dot(mat.themal_cond * grad(self.T),grad(vT)) * dxm
        R_ufl += - mat.vol_heat_cap * self.T_n * vT * dxm
        self.R = R_ufl
        #self.R = R_ufl - self.q_dummy * self.T * vT * dxm
        #self.R = R_ufl - self.q_dummy * self.q_T * vT * dxm

        # derivative
        dR_ufl = derivative(R_ufl, self.T)
        self.dR = dR_ufl
        #self.dR = dR_ufl - self.q_dummy * T_ * vT * dxm

        # setup projector to project continuous funtionspace to quadrature
        #self.project_T = LocalProjector(self.T, q_V, dxm)

        #self.calculate_eps = LocalProjector(eps(d), VQV, dxm)
        #self.calculate_e = LocalProjector(e, VQF, dxm)

        self.assembler = None  #set as default, to check if bc have been added???

    def evaluate_material(self):
        # project stuff (temperautre) onto their quadrature spaces
        pass
        #self.project_T(self.q_T)

        # get the actual values as vector???
        # temperature_vector = self.q_T.vector().get_local()
        #
        # # ... "manually" evaluate_material the material ...
        # self.mat.integrate(eps_flat, e)

        # ... and write the calculated values into their quadrature spaces.
        # from np list/array to quadspace???
        # set_q(self.q_eeq, self.mat.eeq)
        # set_q(self.q_deeq_deps, self.mat.deeq)
        # set_q(self.q_sigma, self.mat.sigma)
        # set_q(self.q_dsigma_deps, self.mat.dsigma_deps)
        # set_q(self.q_dsigma_de, self.mat.dsigma_de)

    def update(self):
        # when or what do I need to update???
        # self.calculate_e(self.q_e)
        # self.mat.update(self.q_e.vector().get_local())
        # set_q(self.q_k, self.mat.kappa)  # just for post processing
        pass

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







# Create mesh and define function space
nx = ny = 4
mesh = UnitSquareMesh(nx, ny)

mat = concrete_material_data() # setting up some basic material things
concrete_problem = concrete_temp_hydration_model(mesh, mat)  #setting up the material problem, with material data and mesh

# Define boundary and initial conditions
t_boundary = 25+273.15  # input in celcius
t_zero = 25+273.15 # input in celcius
t0 = Expression('t_zero', t_zero=t_zero, degree= 0)
u0 = Expression('t_boundary', t_boundary = t_boundary, degree= 1)

def full_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(concrete_problem.V, u0, full_boundary)

concrete_problem.set_bcs(bc)

# TODO is this correctly implemented, is there a better/easier way????
# Initial temp. condition
concrete_problem.T_n = interpolate(t0, concrete_problem.V)

# data for time stepping
#time steps
dt = 60*60# time step
hours = 48
time = hours*60*60          # total simulation time in s

concrete_problem.dt = Constant(dt) # for time integration scheme

t = dt # first time step time

# visualize initial conditions in paraview
pv_file = XDMFFile('output_paraview.xdmf')
pv_file.parameters["flush_output"] = True
pv_file.parameters["functions_share_mesh"] = True
# Plot fields
temperature = Function(concrete_problem.V, name='Temperature')

# paraview export
temperature.assign(concrete_problem.T_n)
pv_file.write(temperature,0) # Save solution to file in XDMF format

# plot data fields
plot_data = [[[],[],[]],[[],[],[]],[[],[],[]]]


solver = NewtonSolver()


#time
while t <= time:
    print('time =', t)
    print('Solving: T')
    solver.solve(concrete_problem, concrete_problem.T.vector())
    # solve temperature

    print(concrete_problem.T.vector()[:])

    # prepare next timestep
    t += dt
    concrete_problem.T_n.assign(concrete_problem.T)
    #concrete_problem.alpha_n.assign(concrete_problem.alpha


    # Output Data
    # Plot displacements
    temperature.assign(concrete_problem.T)
    pv_file.write(temperature,t) # Save solution to file in XDMF format
    #plot points
    plot_data[0][0].append(t)
    plot_data[0][1].append(temperature(0.0,0.0)-273.15)

    plot_data[1][0].append(t)
    plot_data[1][1].append(temperature(0.25,0.25)-273.15)

    plot_data[2][0].append(t)
    plot_data[2][1].append(temperature(0.5,0.5)-273.15)


#plotting stuff
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
ax1.plot(plot_data[0][0], plot_data[0][1])
ax1.plot(plot_data[1][0], plot_data[1][1])
ax1.plot(plot_data[2][0], plot_data[2][1])
ax1.set_title('Temperature')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Temperature')

fig.suptitle('Some points', fontsize=16)

plt.show()

# some copied functions from fenics constitutive

