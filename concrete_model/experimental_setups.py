
import dolfin as df
import numpy as np


from concrete_model.helpers import Parameters

class Experiment:
    def __init__(self, parameters = None):
        # setup of paramter field
        self.p = Parameters()
        # constants
        self.p['zero_C'] = 273.15  # to convert celcius to kelvin input to

        self.p = self.p + parameters

        self.setup()

    def setup(self):
        raise NotImplementedError()


# what is this for (exept for being fun but confusing)
def get_experiment(name, parameters = None):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(parameters)


class ConcreteCubeExperiment(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        # boundary values...
        p['T_0'] = 20  # inital concrete temperature
        p['T_bc1'] = 30  # temperature boundary value 1
        p['T_bc2'] = 50  # temperature boundary value 2
        p['T_bc3'] = 20  # temperature boundary value 3
        p['bc_setting'] = 'full'  # default boundary setting
        p['dim'] = 3  # default boundary setting
        p['mesh_density'] = 10  # default boundary setting
        p['mesh_setting'] = 'left/right'  # default boundary setting
        p = p + parameters
        super().__init__(p)


    def setup(self, bc='full'):
        self.bc = bc  # different boundary settings
        # elements per spacial direction
        n = self.p.mesh_density
        if self.p.dim == 2:
            self.mesh = df.UnitSquareMesh(n, n, self.p.mesh_setting)
        elif self.p.dim == 3:
            self.mesh = df.UnitCubeMesh(n, n, n)
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()

        # self.p['T_0'] = 20 # inital concrete temperature
        # self.p['T_bc1'] = 10 # temperature boundary value 1
        # self.p['T_bc2'] = 50 # temperature boundary value 2
        # self.p['T_bc3'] = 10 # temperature boundary value 3

    def create_temp_bcs(self, V):
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
        T_bc1 = df.Expression('t_boundary', t_boundary=self.p.T_bc1 + self.p.zero_C, degree=0)
        T_bc2 = df.Expression('t_boundary', t_boundary=self.p.T_bc2 + self.p.zero_C, degree=0)
        T_bc3 = df.Expression('t_boundary', t_boundary=self.p.T_bc3 + self.p.zero_C, degree=0)

        temp_bcs = []

        if self.p.bc_setting == 'full':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, full_boundary))
        elif self.p.bc_setting == 'test-setup':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, L_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, U_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc2, R_boundary))
        else:
            raise Exception(
                f'parameter[\'bc_setting\'] = {self.bc_setting} is not implemented as temperature boundary.')

        return temp_bcs

    def create_displ_bcs(self, V):
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

        if self.p.dim == 2:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), U_boundary))
        elif self.p.dim == 3:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0, 0)), U_boundary))

        return displ_bcs


class MinimalCubeExperiment(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        # boundary values...
        p['T_0'] = 20  # inital concrete temperature
        p['dim'] = 2  # default boundary setting
        p['mesh_density'] = 1  # default boundary setting
        p['mesh_setting'] = 'left/right'  # default boundary setting
        p = p + parameters
        super().__init__(p)

    def setup(self, bc='full'):
        self.bc = bc  # different boundary settings
        # elements per spacial direction
        n = self.p.mesh_density
        if self.p.dim == 2:
            self.mesh = df.UnitSquareMesh(n, n, self.p.mesh_setting)
        elif self.p.dim == 3:
            self.mesh = df.UnitCubeMesh(n, n, n)
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()

    def create_temp_bcs(self, V):
        # no temperature boudary!!!

        temp_bcs = []

        return temp_bcs

    def create_displ_bcs(self, V):
        # define surfaces, full, left, right, bottom, top, none
        def full_boundary(x, on_boundary):
            return on_boundary

        # define displacement boundary
        displ_bcs = []

        if self.p.dim == 2:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), full_boundary))
        elif self.p.dim == 3:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0, 0)), full_boundary))

        return displ_bcs


class ConcreteBeamExperiment(Experiment):
    def __init__(self, parameters = None):
        p = Parameters()
        # boundary values...
        p['T_0'] = 20  # inital concrete temperature
        p['T_bc1'] = 30  # temperature boundary value 1
        p['T_bc2'] = 50  # temperature boundary value 2
        p['T_bc3'] = 20  # temperature boundary value 3
        p['l'] = 5 # m length
        p['h'] = 1 # height
        p['bc_setting'] = 'full' # default boundary setting
        p = p + parameters
        super().__init__(p)

    def setup(self,bc = 'full', dim = 2):
        self.bc = bc # different boundary settings
        # elements per spacial direction
        n = 20
        # TODO 3D beam!?!
        if dim == 2:
            self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.p.l, self.p.h) \
                                     , int(n * self.p.l), int(n * self.p.h), diagonal='right')
        else:
            print(f'wrong dimension {dim} for problem setup')
            exit()

        # self.p['T_0'] = 20 # inital concrete temperature
        # self.p['T_bc1'] = 10 # temperature boundary value 1
        # self.p['T_bc2'] = 50 # temperature boundary value 2
        # self.p['T_bc3'] = 10 # temperature boundary value 3


    def create_temp_bcs(self,V):
        # define surfaces, full, left, right, bottom, top, none
        def full_boundary(x, on_boundary):
            return on_boundary
        def L_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 0)
        def R_boundary(x, on_boundary):
            return on_boundary and df.near(x[0],  self.p.l)
        def U_boundary(x, on_boundary):
            return on_boundary and df.near(x[1], 0)
        def O_boundary(x, on_boundary):
            return on_boundary and df.near(x[1],  self.p.h)
        def empty_boundary(x, on_boundary):
            return None

        # Temperature boundary conditions
        T_bc1 = df.Expression('t_boundary', t_boundary=self.p.T_bc1+self.p.zero_C, degree=0)
        T_bc2 = df.Expression('t_boundary', t_boundary=self.p.T_bc2+self.p.zero_C, degree=0)
        T_bc3 = df.Expression('t_boundary', t_boundary=self.p.T_bc3+self.p.zero_C, degree=0)

        temp_bcs = []

        if self.p.bc_setting == 'full':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, full_boundary))
        elif self.p.bc_setting == 'left-right':
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
            return df.near(x[0], self.p.l) and df.near(x[1], 0)

        def L_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], 0)
        def R_boundary(x, on_boundary):
            return on_boundary and df.near(x[0], self.p.l)


        # define displacement boundary
        displ_bcs = []
        displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), left_support, method='pointwise'))
        displ_bcs.append(df.DirichletBC(V.sub(1), df.Constant(0), right_support, method='pointwise'))

        #displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), L_boundary))
        #displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), R_boundary))

        return displ_bcs
        
  
