from __future__ import print_function
#from fenics import *
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self):
        self.data = {}
        self.setup()

    def add_sensor_data(self, sensor, data, ts=[1.0]):
        self.data[sensor] = (data, ts)

    def refine(self, N=1):
        """
        Refines the mesh `N` times.
        """
        for _ in range(N):
            self.mesh = df.refine(self.mesh)

    def setup(self):
        raise NotImplementedError()




class Parameters(dict):
    """
    Dict that also allows to access the parameter
        p["parameter"]
    via the matching attribute
        p.parameter
    to make access shorter
    """
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def __add__(self, other):
        return Parameters({**self, **other})


class ConcreteCubeExperiment(Experiment):
    def __init__(self, parameters = None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        # boundary values...
        p['T_0'] = 20  # inital concrete temperature
        p['T_bc1'] = 30  # temperature boundary value 1
        p['T_bc2'] = 50  # temperature boundary value 2
        p['T_bc3'] = 20  # temperature boundary value 3
        p['bc_setting'] = 'full' # default boundary setting
        p['dim'] = 3 # default boundary setting
        p['mesh_density'] = 10 # default boundary setting
        p['mesh_setting'] = 'crossed' # default boundary setting

        # add and override input paramters
        if parameters == None:
            self.parameters = p
        else:
            self.parameters = p + parameters

        super().__init__()

    def setup(self,bc = 'full'):
        self.bc = bc # different boundary settings
        # elements per spacial direction
        n = self.parameters.mesh_density
        if self.parameters.dim == 2:
            self.mesh = df.UnitSquareMesh(n, n,  self.parameters.mesh_setting)
        elif self.parameters.dim == 3:
            self.mesh = df.UnitCubeMesh(n, n, n)
        else:
            print(f'wrong dimension {self.parameters.dim} for problem setup')
            exit()

        # define paramters???
        self.zero_C = 273.15 # to convert celcius to kelvin input to
        # self.parameters['T_0'] = 20 # inital concrete temperature
        # self.parameters['T_bc1'] = 10 # temperature boundary value 1
        # self.parameters['T_bc2'] = 50 # temperature boundary value 2
        # self.parameters['T_bc3'] = 10 # temperature boundary value 3


    def create_temp_bcs(self,V):
        # define surfaces, full, left, right, bottom, top, none
        def full_boundary(x, on_boundary):
            return on_boundary
        def L_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 0)
        def R_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 1)
        def U_boundary(x, on_boundary):
            return on_boundary and df.near(x[1], 0)
        def O_boundary(x, on_boundary):
            return on_boundary and df.near(x[1], 1)
        def empty_boundary(x, on_boundary):
            return None

        # Temperature boundary conditions
        T_bc1 = df.Expression('t_boundary', t_boundary=self.parameters.T_bc1+self.zero_C, degree=0)
        T_bc2 = df.Expression('t_boundary', t_boundary=self.parameters.T_bc2+self.zero_C, degree=0)
        T_bc3 = df.Expression('t_boundary', t_boundary=self.parameters.T_bc3+self.zero_C, degree=0)

        temp_bcs = []

        if self.parameters.bc_setting == 'full':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, full_boundary))
        elif self.parameters.bc_setting == 'left-right':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc2, L_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc2, U_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc3, R_boundary))
        else:
            raise Exception(f'parameter[\'bc_setting\'] = {self.bc_setting} is not implemented as temperature boundary.')

        return temp_bcs


    def create_displ_bcs(self,V):
        # define surfaces, full, left, right, bottom, top, none
        def full_boundary(x, on_boundary):
            return on_boundary
        def L_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 0)
        def R_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 1)
        def U_boundary(x, on_boundary):
            return on_boundary and df.near(x[1], 0)
        def O_boundary(x, on_boundary):
            return on_boundary and df.near(x[1], 1)
        def empty_boundary(x, on_boundary):
            return None

        # define displacement boundary
        displ_bcs = []

        if self.parameters.dim == 2:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), U_boundary))
        elif self.parameters.dim == 3:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0, 0)), U_boundary))

        return displ_bcs


class ConcreteBeamExperiment(Experiment):
    def __init__(self, parameters = None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        # boundary values...
        p['T_0'] = 20  # inital concrete temperature
        p['T_bc1'] = 30  # temperature boundary value 1
        p['T_bc2'] = 50  # temperature boundary value 2
        p['T_bc3'] = 20  # temperature boundary value 3
        p['l'] = 5 # m length
        p['h'] = 1 # height
        p['bc_setting'] = 'full' # default boundary setting


        # add and override input paramters
        if parameters == None:
            self.parameters = p
        else:
            self.parameters = p + parameters

        super().__init__()

    def setup(self,bc = 'full', dim = 2):
        self.bc = bc # different boundary settings
        # elements per spacial direction
        n = 20
        if dim == 2:
            self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.parameters.l, self.parameters.h) \
                                     , int(n * self.parameters.l), int(n * self.parameters.h), diagonal='right')
        else:
            print(f'wrong dimension {dim} for problem setup')
            exit()

        # define paramters???
        self.zero_C = 273.15 # to convert celcius to kelvin input to
        # self.parameters['T_0'] = 20 # inital concrete temperature
        # self.parameters['T_bc1'] = 10 # temperature boundary value 1
        # self.parameters['T_bc2'] = 50 # temperature boundary value 2
        # self.parameters['T_bc3'] = 10 # temperature boundary value 3


    def create_temp_bcs(self,V):
        # define surfaces, full, left, right, bottom, top, none
        def full_boundary(x, on_boundary):
            return on_boundary
        def L_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 0)
        def R_boundary(x, on_boundary):
            return on_boundary and df.near(x[0],  self.parameters.l)
        def U_boundary(x, on_boundary):
            return on_boundary and df.near(x[1], 0)
        def O_boundary(x, on_boundary):
            return on_boundary and df.near(x[1],  self.parameters.h)
        def empty_boundary(x, on_boundary):
            return None

        # Temperature boundary conditions
        T_bc1 = df.Expression('t_boundary', t_boundary=self.parameters.T_bc1+self.zero_C, degree=0)
        T_bc2 = df.Expression('t_boundary', t_boundary=self.parameters.T_bc2+self.zero_C, degree=0)
        T_bc3 = df.Expression('t_boundary', t_boundary=self.parameters.T_bc3+self.zero_C, degree=0)

        temp_bcs = []

        if self.parameters.bc_setting == 'full':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, full_boundary))
        elif self.parameters.bc_setting == 'left-right':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc2, L_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc3, R_boundary))
        else:
            raise Exception(f'parameter[\'bc_setting\'] = {self.bc_setting} is not implemented as temperature boundary.')

        return temp_bcs


    def create_displ_bcs(self,V):
        # define surfaces, full, left, right, bottom, top, none
        def left_support(x, on_boundary):
            return df.near(x[0], 0) and df.near(x[1], 0)
        def right_support(x, on_boundary):
            return df.near(x[0], self.parameters.l) and df.near(x[1], 0)

        def L_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 0)
        def R_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], self.parameters.l)


        # define displacement boundary
        displ_bcs = []
        displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left_support, method='pointwise'))
        displ_bcs.append(df.DirichletBC(V.sub(1), df.Constant(0), right_support, method='pointwise'))

        #displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), L_boundary))
        #displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), R_boundary))

        return displ_bcs






# what is this for (exept for being fun but confusing)
def get_experiment(name, parameters = None):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(parameters)



class Sensor:
    def measure(self, u):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class DisplacementSensor(Sensor):
    def __init__(self, where):
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get displacements
        #print(type(problem.mechanics_problem.u(self.where)))
        self.data.append(np.concatenate(([t],problem.mechanics_problem.u(self.where))))


class TemperatureSensor(Sensor):
    def __init__(self, where):
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get displacements
        #print(type(problem.mechanics_problem.u(self.where)))
        #T = 42
        T = problem.temperature_problem.T(self.where) - problem.temperature_problem.zero_C
        self.data.append([t,T])

class MaxTemperatureSensor(Sensor):
    def __init__(self):
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        max_T = np.amax(problem.temperature_problem.T.vector().get_local()) - problem.temperature_problem.zero_C

        self.data.append([t,max_T])

class DOHSensor(Sensor):
    def __init__(self, where):
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get DOH
        alpha_projected = df.project(problem.temperature_problem.q_alpha, problem.temperature_problem.visu_space)
        alpha = alpha_projected(self.where)
        self.data.append([t,alpha])

class MinDOHSensor(Sensor):
    def __init__(self):
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get min DOH
        min_DOH = np.amin(problem.temperature_problem.q_alpha.vector().get_local())
        self.data.append([t,min_DOH])

class DOHHomogeneitySensor(Sensor):
    # gives a measure of degree of homogeneity
    # computes average DOH (only applicable if all elements same size)
    # computes the square of the difference of each element and computes the sum
    def __init__(self):
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get data
        alpha_list = problem.temperature_problem.q_alpha.vector().get_local()
        ave_DOH = np.average(alpha_list)
        error = ((alpha_list-ave_DOH)**2).sum()
        self.data.append([t,ave_DOH, error])