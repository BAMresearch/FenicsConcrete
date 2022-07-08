

from fenics_concrete.experimental_setups.experiment import Experiment
from fenics_concrete.helpers import Parameters
import dolfin as df


class ConcreteBeamExperiment(Experiment):
    def __init__(self, parameters = None):
        p = Parameters()
        # boundary values...
        p['T_0'] = 20  # inital concrete temperature
        p['T_bc1'] = 30  # temperature boundary value 1
        p['T_bc2'] = 50  # temperature boundary value 2
        p['T_bc3'] = 20  # temperature boundary value 3
        p['length'] = 5 # m length
        p['height'] = 1 # height
        p['width'] = 0.8 # width
        p['mesh_density'] = 4  # number of elements in vertical dirrecton, the others are set accordingly
        p['bc_setting'] = 'full' # default boundary setting
        p = p + parameters
        super().__init__(p)

        # initialize variable top_displacement
        self.displ_load = df.Constant(0.0)  # applied via fkt: apply_displ_load(...)


    def setup(self):
        # computing the number of elements in each direcction
        n_height = int(self.p['mesh_density'])
        n_width = int(n_height/self.p.height*self.p.width)
        n_length = int(n_height/self.p.height*self.p.length)
        if (n_length % 2) != 0: # check for odd number
            n_length += 1 # n_length must be even for loading example

        if self.p.dim == 2:
             self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.p.length, self.p.height),
                                          n_length,
                                          n_height, diagonal='right')
        elif self.p.dim == 3:
            self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(self.p.length, self.p.width, self.p.height),
                                   n_length, n_width, n_height)
        else:
            raise Exception(f'wrong dimension {self.p.dim} for problem setup')


    def create_temp_bcs(self,V):

        # Temperature boundary conditions
        T_bc1 = df.Expression('t_boundary', t_boundary=self.p.T_bc1+self.p.zero_C, degree=0)
        T_bc2 = df.Expression('t_boundary', t_boundary=self.p.T_bc2+self.p.zero_C, degree=0)
        T_bc3 = df.Expression('t_boundary', t_boundary=self.p.T_bc3+self.p.zero_C, degree=0)

        temp_bcs = []

        if self.p.bc_setting == 'full':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc1, self.boundary_full()))
        elif self.p.bc_setting == 'left-right':
            # bc.append(DirichletBC(temperature_problem.V, T_bc, full_boundary))
            temp_bcs.append(df.DirichletBC(V, T_bc2, self.boundary_left()))
            temp_bcs.append(df.DirichletBC(V, T_bc3, self.boundary_right()))
        else:
            raise Exception(f'parameter[\'bc_setting\'] = {self.p.bc_setting} is not implemented as temperature boundary.')

        return temp_bcs


    def create_displ_bcs(self,V):
        if self.p.dim == 2:
            dir_id = 1
            fixed_bc = df.Constant((0, 0))
        elif self.p.dim == 3:
            dir_id = 2
            fixed_bc = df.Constant((0, 0, 0))

        # define surfaces, full, left, right, bottom, top, none
        def left_support(x, on_boundary):
            return df.near(x[0], 0) and df.near(x[dir_id], 0)
        def right_support(x, on_boundary):
            return df.near(x[0], self.p.length) and df.near(x[dir_id], 0)
        def center_top(x, on_boundary):
            return df.near(x[0], self.p.length/2) and df.near(x[dir_id], self.p.height)



        # define displacement boundary
        displ_bcs = []

        displ_bcs.append(df.DirichletBC(V, fixed_bc, left_support, method='pointwise'))
        displ_bcs.append(df.DirichletBC(V.sub(dir_id), df.Constant(0), right_support, method='pointwise'))
        displ_bcs.append(df.DirichletBC(V.sub(dir_id), self.displ_load, center_top, method='pointwise'))

        return displ_bcs


    def apply_displ_load(self, displacement_load):
        """Updates the applied displacement load

        Parameters
        ----------
        top_displacement : float
            Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression
        """

        self.displ_load.assign(df.Constant(displacement_load))
