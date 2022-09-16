from fenics_concrete.experimental_setups.experiment import Experiment
from fenics_concrete.helpers import Parameters
import dolfinx as df
from mpi4py import MPI
import numpy as np
from petsc4py.PETSc import ScalarType

class DomainDefinition(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        # boundary values...
        p['bc_setting'] = 'full'  # default boundary setting
        p['degree'] = 2  # polynomial degree
        p['dim'] = 2  # default boundary setting
        p = p + parameters
        super().__init__(p)

    def setup(self, bc='full'):
        self.bc = bc  # different boundary settings

        # elements per spatial direction
        if self.p.dim == 2:
            #self.mesh = df.UnitSquareMesh(n, n, self.p.mesh_setting)
            self.mesh = df.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (self.p.length, self.p.breadth)), n=(self.p.num_elements_length, self.p.num_elements_breadth),
                            cell_type=df.mesh.CellType.quadrilateral)
        elif self.p.dim == 3:
            #self.mesh = df.UnitCubeMesh(n, n, n)
            self.mesh = df.mesh.create_box(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (self.p.length, self.p.breadth, self.p.height)), n=(self.p.num_elements_length, self.p.num_elements_breadth, self.p.num_elements_height),
                            cell_type=df.mesh.CellType.hexahedron)
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()

        # define function space ets.
        self.V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", self.p.degree)) # 2 for quadratic elements

        # boundary conditions only after function space
        self.bcs = self.create_displ_bcs(self.V)

    def create_displ_bcs(self, V):
        # define displacement boundary

        def clamped_boundary(x):          # fenics will individually call this function for every node and will note the true or false value.
            return np.isclose(x[0], 0)

        displ_bcs = []
        if self.p.dim == 2:
            #displ_bcs.append(df.fem.DirichletBC(V, df.Constant((0, 0)), self.boundary_left()))
            displ_bcs.append(df.fem.dirichletbc(np.array([0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(V, clamped_boundary), V))
            
        elif self.p.dim == 3:
            #displ_bcs.append(df.fem.DirichletBC(V, df.Constant((0, 0, 0)),  self.boundary_left()))
            displ_bcs.append(df.fem.dirichletbc(np.array([0, 0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(V, clamped_boundary), V))

        return displ_bcs

    #def apply_load_bc(self, V):

