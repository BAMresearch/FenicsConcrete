

from concrete_model.experimental_setups.template_experiment import Experiment
from concrete_model.helpers import Parameters
import dolfin as df


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
        
  
