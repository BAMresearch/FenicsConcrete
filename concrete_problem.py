from __future__ import print_function
import dolfin as df
import numpy as np
import scipy.optimize
import time
import concrete_experiment as concrete_experiment

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
        dv = df.TrialFunction(V)
        v_ = df.TestFunction(V)
        a_proj = df.inner(dv, v_) * dxm
        b_proj = df.inner(expr, v_) * dxm
        self.solver = df.LocalSolver(a_proj, b_proj)
        self.solver.factorize()

    def __call__(self, u):
        """
        u:
            function that is filled with the solution of the projection
        """
        self.solver.solve_local_rhs(u)

# full concrete model, including hydration-temperate and mechanics, including calls to solve etc.
class ConcreteThermoMechanical():
    def __init__(self, experiment, parameters):
        # ideas is, (to be compatible with our current fenics module plans) have "experiment" as input
        # currently this is the mesh, and later the calls to bc functions. TODO:
        # TODO: define global fields here
        #       - alpha, V
        #       - etc...
        #       fix error related to quad degree (add as "global")
        #       add mechanics paramter for fc and fct!!!
        #       check logic about defined functions, classes etc...
        #       add some sensor output options???
        self.experiment = experiment
        self.parameters = parameters
        self.setup()

    def setup(self, pv_name = 'pv_output_full'):
        # setting up the two nonlinear problems
        self.temperature_problem = ConcreteTempHydrationModel(self.experiment.mesh, self.parameters, pv_name = pv_name)
        # TODO paramtersetup not jet "perfect"
        # here I "pass on the parameters from temperature to mechanics problem.."
        self.mechanics_problem = ConcreteMechanicsModel(self.experiment.mesh, self.temperature_problem.mat, pv_name = pv_name)
        # coupling of the output files
        self.mechanics_problem.pv_file = self.temperature_problem.pv_file

        #initialize concrete temperature as given in experimental setup
        self.set_inital_T(self.experiment.parameters.T_0)

        #setting bcs
        self.mechanics_problem.set_bcs(self.experiment.create_displ_bcs(self.mechanics_problem.V))
        self.temperature_problem.set_bcs(self.experiment.create_temp_bcs(self.temperature_problem.V))

        # setting up the solvers
        self.temperature_solver = df.NewtonSolver()
        self.temperature_solver.parameters['absolute_tolerance'] = 1e-9
        self.temperature_solver.parameters['relative_tolerance'] = 1e-8

        self.mechanics_solver = df.NewtonSolver()
        self.mechanics_solver.parameters['absolute_tolerance'] = 1e-9
        self.mechanics_solver.parameters['relative_tolerance'] = 1e-8

    def solve(self):

        print('Solving: T')
        self.temperature_solver.solve(self.temperature_problem, self.temperature_problem.T.vector())

        # set current DOH for computation of Young's modulus
        self.mechanics_problem.q_alpha = self.temperature_problem.q_alpha

        print('Solving: u')
        self.mechanics_solver.solve(self.mechanics_problem, self.mechanics_problem.u.vector())

        # history update
        self.temperature_problem.update_history()


    def pv_plot(self,t = 0):
        self.temperature_problem.pv_plot(t=t)
        self.mechanics_problem.pv_plot(t=t)


    def set_inital_T(self,T):
        # TODO, somehow check that this is initialized
        self.temperature_problem.set_initial_T(T)
        self.flag_T0 = True

    def set_timestep(self,dt):
        self.temperature_problem.set_timestep(dt)


class ConcreteTempHydrationModel(df.NonlinearProblem):
    def __init__(self, mesh, mat, pv_name = 'temp_output', **kwargs):
        df.NonlinearProblem.__init__(self) # apparently required to initialize things
        # todo how/where to deal with constants...
        # maybe a constants class/object to collect them all???
        self.zero_C = 273.15
        self.igc = 8.3145  # ideal gas constant [JK −1 mol −1 ]

        #setup initial material paramters
        p = concrete_experiment.Parameters()
        # Material parameter for concrete model with temperature and hydration
        p['density'] = 2350  # in kg/m^3 density of concrete
        p['density_binder'] = 1440  # in kg/m^3 density of the binder
        p['themal_cond'] = 2.0  # effective thermal conductivity, approx in Wm^-3K^-1, concrete!
        # self.specific_heat_capacity = 9000  # effective specific heat capacity in J kg⁻1 K⁻1
        p['vol_heat_cap'] = 2.4e6  # volumetric heat cap J/(m3 K)
        p['b_ratio'] = 0.2  # volume percentage of binder
        p['Q_pot'] = 500e3  # potential heat per weight of binder in J/kg
        # p['Q_inf'] = self.Q_pot * self.density_binder * self.b_ratio  # potential heat per concrete volume in J/m3
        p['B1'] = 2.916E-4  # in 1/s
        p['B2'] = 0.0024229  # -
        p['eta'] = 5.554  # something about diffusion
        p['alpha_max'] = 0.875  # also possible to approximate based on equation with w/c
        p['E_act'] = 5653 * self.igc  # activation energy in Jmol^-1
        p['T_ref'] = 25  # reference temperature in degree celsius
        # setting for temperature adjustment
        # option: 'exponential' and 'off'
        p['temp_adjust_law'] = 'exponential'

        # add parameters to other input values

        if mat == None:
            self.mat = p
        else:
            self.mat = mat + p # object with material data, parameters, material functions etc...


        #initialize possible paraview output
        self.pv_file = df.XDMFFile( pv_name + '.xdmf')
        self.pv_file.parameters["flush_output"] = True
        self.pv_file.parameters["functions_share_mesh"] = True
        # function space for single value per element, required for plot of quadrature space values
        self.visu_space = df.FunctionSpace(mesh, "DG", 0)

        #initialize timestep, musst be reset using .set_timestep(dt)
        self.dt = 0
        self.dt_form = df.Constant(self.dt)

        # TODO why does q_deg = 2 throw errors???
        q_deg = 1

        metadata = {"quadrature_degree": q_deg, "quadrature_scheme": "default"}
        dxm = df.dx(metadata=metadata)

        # solution field
        self.V = df.FunctionSpace(mesh, 'P', q_deg)

        # generic quadrature function space
        cell = mesh.ufl_cell()
        q = "Quadrature"
        quadrature_element = df.FiniteElement(q, cell, degree=q_deg, quad_scheme="default")
        q_V = df.FunctionSpace(mesh, quadrature_element)

        # quadrature functions
        self.q_T = df.Function(q_V, name="temperature")
        self.q_alpha = df.Function(q_V, name="degree of hydration")
        self.q_alpha_n = df.Function(q_V, name="degree of hydration last time step")
        self.q_delta_alpha = df.Function(q_V, name="inrease in degree of hydration")
        self.q_ddalpha_dT = df.Function(q_V, name="derivative of delta alpha wrt temperature")

        # empfy list for newton iteration to compute delta alpha using the last value as starting point
        self.delta_alpha_n_list = np.full(np.shape(self.q_alpha_n.vector().get_local() ), 0.2)

        # scalars for the analysis of the heat of hydration
        self.alpha = 0
        self.delta_alpha = 0

        # Define variational problem
        self.T = df.Function(self.V)  # temperature
        self.T_n = df.Function(self.V)  # overwritten later...
        T_ = df.TrialFunction(self.V) # temperature
        vT = df.TestFunction(self.V)

        # normal form
        R_ufl =  df.Constant(self.mat.vol_heat_cap) * (self.T) * vT * dxm
        R_ufl += self.dt_form * df.dot( df.Constant(self.mat.themal_cond) * df.grad(self.T),df.grad(vT)) * dxm
        R_ufl += -  df.Constant(self.mat.vol_heat_cap) * self.T_n * vT * dxm
        # quadrature point part


        self.R = R_ufl - df.Constant(self.mat.Q_pot * self.mat.density_binder * self.mat.b_ratio) * self.q_delta_alpha * vT * dxm

        # derivative
        # normal form
        dR_ufl = df.derivative(R_ufl, self.T)
        # quadrature part
        self.dR = dR_ufl - df.Constant(self.mat.Q_pot * self.mat.density_binder * self.mat.b_ratio) * self.q_ddalpha_dT * T_ * vT * dxm

        # setup projector to project continuous funtionspace to quadrature
        self.project_T = LocalProjector(self.T, q_V, dxm)

        self.assembler = None  #set as default, to check if bc have been added???


    def delta_alpha_fkt(self,delta_alpha, alpha_n, T):
        return delta_alpha - self.dt * self.affinity(delta_alpha, alpha_n) * self.temp_adjust(T)


    def delta_alpha_prime(self,delta_alpha, alpha_n, T):
        return 1 - self.dt * self.daffinity_ddalpha(delta_alpha, alpha_n) * self.temp_adjust(T)


    def get_heat_of_hydration(self,tmax,T):
        t = 0
        time = []
        heat = []
        alpha_list = []
        alpha = 0
        delta_alpha = 0.2
        while t <= tmax:
            time.append(t)
            # compute delta_alpha
            delta_alpha = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha, T+self.zero_C),
                                  fprime=self.delta_alpha_prime, x0=delta_alpha)
            # update alpha
            alpha = delta_alpha + alpha
            # save heat of hydration
            alpha_list.append(delta_alpha)
            heat.append(alpha*self.mat.Q_pot * self.mat.density_binder * self.mat.b_ratio)

            # timeupdate
            t = t+self.dt


        return np.asarray(time)/60/60, np.asarray(heat)/1000, np.asarray(alpha_list)


    def evaluate_material(self):
        # project temperautre onto quadrature spaces
        self.project_T(self.q_T)

        # convert quadrature spaces to numpy vector
        temperature_list = self.q_T.vector().get_local()
        alpha_n_list = self.q_alpha_n.vector().get_local()

        # solve for alpha at each quadrature point
        # here the newton raphson method of the scipy package is used
        # the zero value of the delta_alpha_fkt is found for each entry in alpha_n_list is found. the corresponding temparature
        # is given in temperature_list and as starting point the value of last step used from delta_alpha_n
        delta_alpha_list = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha_n_list, temperature_list),fprime=self.delta_alpha_prime, x0=self.delta_alpha_n_list)

        # I dont trust the algorithim!!! check if only applicable results are obtained
        if np.any(delta_alpha_list<0.0):
            # TODO: better error message ;)
            print('AAAAAAHHHH, negative delta alpha!!!!')
            exit()

        # save the delta alpha for next iteration as starting guess
        self.delta_alpha_n_list = delta_alpha_list

        # compute current alpha
        alpha_list = alpha_n_list + delta_alpha_list
        # compute derivative of delta alpha with respect to temperature for rhs
        ddalpha_dT_list = self.dt * self.affinity(alpha_list, alpha_n_list)* self.temp_adjust_tangent(temperature_list)

        # project lists onto quadrature spaces
        set_q(self.q_alpha, alpha_list)
        set_q(self.q_delta_alpha, delta_alpha_list)
        set_q(self.q_ddalpha_dT, ddalpha_dT_list)


    def update_history(self):
        self.T_n.assign(self.T) # save temparature field
        self.q_alpha_n.assign(self.q_alpha) # save alpha field


    def set_timestep(self, dt):
        self.dt = dt
        self.dt_form.assign(df.Constant(self.dt))


    def set_initial_T(self,T):
        #
        # set initial temperature, in kelvin
        T0 = df.Expression('t_zero', t_zero=T+self.zero_C, degree=0)
        self.T_n.interpolate(T0)
        self.T.interpolate(T0)


    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = df.SystemAssembler(self.dR, self.R, bcs)


    def F(self, b, x):
        if self.dt <= 0:
            raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)

    def pv_plot(self,t = 0):
        # paraview export

        # temperature plot
        T_plot = df.project(self.T, self.V)
        T_plot.rename("Temperature","test string, what does this do??")  # TODO: what does the second string do?
        self.pv_file.write(T_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        # degree of hydration plot
        alpha_plot = df.project(self.q_alpha, self.visu_space)
        alpha_plot.rename("DOH","test string, what does this do??")  # TODO: what does the second string do?
        self.pv_file.write(alpha_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

    def temp_adjust(self, T):
        val = 1
        if self.mat.temp_adjust_law == 'exponential':
            val = np.exp(-self.mat.E_act / self.igc * (1 / T - 1 / (self.mat.T_ref + self.zero_C)))
        elif self.mat.temp_adjust_law == 'off':
            pass
        else:
            # TODO throw correct error
            print(f'Warning: Incorrect temp_adjust_law {self.mat.temp_adjust_law} given')
            print('*******  Only "exponential" and "off" implemented')
        return val

        # derivative of the temperature adjustment factor with respect to the temperature
    def temp_adjust_tangent(self, T):
        val = 0
        if self.mat.temp_adjust_law == 'exponential':
            val = self.mat.E_act / self.igc / T ** 2
        return val

    # affinity function
    def affinity(self, delta_alpha, alpha_n):
        affinity = self.mat.B1 * (self.mat.B2 / self.mat.alpha_max + delta_alpha + alpha_n) * (
                    self.mat.alpha_max - (delta_alpha + alpha_n)) * np.exp(
            -self.mat.eta * (delta_alpha + alpha_n) / self.mat.alpha_max)
        return affinity

    # derivative of affinity with respect to delta alpha
    def daffinity_ddalpha(self, delta_alpha, alpha_n):
        affinity_prime = self.mat.B1 * np.exp(-self.mat.eta * (delta_alpha + alpha_n) / self.mat.alpha_max) * (
                    (self.mat.alpha_max - (delta_alpha + alpha_n)) * (
                        self.mat.B2 / self.mat.alpha_max + (delta_alpha + alpha_n)) * (
                                -self.mat.eta / self.mat.alpha_max) - self.mat.B2 / self.mat.alpha_max - 2 * (
                                delta_alpha + alpha_n) + self.mat.alpha_max)
        return affinity_prime






class ConcreteMechanicsModel(df.NonlinearProblem):
    def __init__(self, mesh, mat, pv_name = 'mechanics_output', **kwargs):
        df.NonlinearProblem.__init__(self) # apparently required to initialize things
        # constants
        self.g = 9.81  # graviational acceleration in m/s²

        # object with material data, parameters, material functions etc...
        p = concrete_experiment.Parameters()
        ### paramters for mechanics problem
        p['E_28'] = 2000000  # Youngs Modulus N/m2 or something... TODO: check units!
        p['nu'] = 0.2  # Poissons Ratio

        # required paramters for alpha to E mapping
        p['alpha_t'] = 0.2
        p['alpha_0'] = 0.05
        p['exp'] = 0.6

        if mat == None:
            self.mat = p
        else:
            self.mat = mat + p # object with material data, parameters, material functions etc...

        #initialize possible paraview output
        self.pv_file = df.XDMFFile( pv_name + '.xdmf')
        self.pv_file.parameters["flush_output"] = True
        self.pv_file.parameters["functions_share_mesh"] = True
        # function space for single value per element, required for plot of quadrature space values
        self.visu_space = df.FunctionSpace(mesh, "DG", 0)
        self.visu_space_T = df.TensorFunctionSpace(mesh, "DG", 0)

        #initialize timestep, musst be reset using .set_timestep(dt)
        #self.dt = 0
        #self.dt_form = Constant(self.dt)

        # TODO why does q_deg = 2 throw errors???
        q_deg = 1

        metadata = {"quadrature_degree": q_deg, "quadrature_scheme": "default"}
        dxm = df.dx(metadata=metadata)

        # solution field
        #self.V = VectorFunctionSpace(mesh, 'P', q_deg)
        self.V = df.VectorFunctionSpace(mesh, 'Lagrange', q_deg)

        # generic quadrature function space
        cell = mesh.ufl_cell()
        q = "Quadrature"
        quadrature_element = df.FiniteElement(q, cell, degree=q_deg, quad_scheme="default")
        q_V = df.FunctionSpace(mesh, quadrature_element)

        # quadrature functions
        self.q_E = df.Function(q_V, name="youngs modulus")
        self.q_alpha = df.Function(q_V, name="degree of hydration")
        # initialize degree of hydration to 1, in case machanics module is run without hydration coupling
        self.q_alpha.vector()[:] = 1


        # Define variational problem
        self.u = df.Function(self.V)  # displacement
        v = df.TestFunction(self.V)


        # Elasticity parameters without multiplication with E
        x_mu = 1.0 / (2.0 * (1.0 + self.mat.nu))
        x_lambda = 1.0 * self.mat.nu / ((1.0 + self.mat.nu) * (1.0 - 2.0 * self.mat.nu))

        # Stress computation for linear elastic problem without multiplication with E
        def x_sigma(v):
            return 2.0 * x_mu * df.sym(df.grad(v)) + x_lambda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))

        # Volume force
        f = df.Constant((0, -self.g * self.mat.density))
        #f = Constant((10, 0))

        #E.assign(Constant(alpha * E_max))
        # solve the mechanics problem
        #self.E = Constant(self.mat.E_28)
        #E.assign(Constant(self.alpha*self.mat.E_28))
        # normal form

        # TODO: why is this working???, is q_E treated as a "constant"?
        self.sigma_ufl = self.q_E*x_sigma(self.u)
        R_ufl =  self.q_E*df.inner(x_sigma(self.u), df.sym(df.grad(v)))  * dxm
        R_ufl += - df.inner(f, v) * dxm # add volumetric force, aka gravity (in this case)
        # quadrature point part
        self.R = R_ufl #- Constant(mat.Q_inf) * self.q_delta_alpha * vT * dxm

        # derivative
        # normal form
        dR_ufl = df.derivative(R_ufl, self.u)
        # quadrature part
        self.dR = dR_ufl #- Constant(mat.Q_inf) * self.q_ddalpha_dT * T_ * vT * dxm

        # setup projector to project continuous funtionspace to quadrature
        #self.project_T = LocalProjector(self.T, q_V, dxm)

        self.assembler = None  #set as default, to check if bc have been added???






    def E_fkt(self,alpha):
        if alpha < self.mat.alpha_t:
            E = self.mat.E_28*alpha/self.mat.alpha_t*((self.mat.alpha_t-self.mat.alpha_0)/(1-self.mat.alpha_0))**self.mat.exp
        else:
            E = self.mat.E_28*((alpha-self.mat.alpha_0)/(1-self.mat.alpha_0))**self.mat.exp
        return E

    def evaluate_material(self):
        # convert quadrature spaces to numpy vector
        alpha_list = self.q_alpha.vector().get_local()

        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_list = E_fkt_vectorized(alpha_list)

        # # project lists onto quadrature spaces
        set_q(self.q_E, E_list)
        pass


    def update_history(self):
        #self.T_n.assign(self.T) # save temparature field
        #self.q_alpha_n.assign(self.q_alpha) # save alpha field
        pass


    def set_timestep(self, dt):
        self.dt = dt
        self.dt_form.assign(df.Constant(self.dt))



    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = df.SystemAssembler(self.dR, self.R, bcs)


    def F(self, b, x):
        #if self.dt <= 0:
        #    raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)

    def pv_plot(self,t = 0):
        # paraview export

        # temperature plot
        u_plot = df.project(self.u, self.V)
        u_plot.rename("Displacement","test string, what does this do??")  # TODO: what does the second string do?
        self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        # stress plot???

        #stress_plot = project(self.q_alpha, self.visu_space)
        # some ufl thing....
        # Stress computation for linear elastic problem without multiplication with E

        # Elasticity parameters without multiplication with E
        x_mu = 1.0 / (2.0 * (1.0 + self.mat.nu))
        x_lambda = 1.0 * self.mat.nu / ((1.0 + self.mat.nu) * (1.0 - 2.0 * self.mat.nu))
        def x_sigma(v):
             return 2.0 * x_mu * df.sym(df.grad(v)) + x_lambda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))
        # stress = assemble(x_sigma(self.u))

        #Some_plot = project(x_sigma(self.u), self.visu_space_T)
        sigma_plot = df.project(self.sigma_ufl, self.visu_space_T)


        E_plot = df.project(self.q_E, self.visu_space)
        #print(stress)

        # youngsmodulus??
        #alpha_plot = project(self.q_alpha, self.visu_space)
        E_plot.rename("Young's Modulus","test string, what does this do??")  # TODO: what does the second string do?
        sigma_plot.rename("Stress","test string, what does this do??")  # TODO: what does the second string do?
        self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        pass
