from concrete_model.experimental_setups.template_experiment import Experiment
from concrete_model.helpers import Parameters
import dolfin as df

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