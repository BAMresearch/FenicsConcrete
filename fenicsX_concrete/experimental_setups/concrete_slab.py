from fenicsX_concrete.experimental_setups.experiment import Experiment
from fenicsX_concrete.helpers import Parameters
import dolfinx as df
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType

class concreteSlabExperiment(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        # boundary values...
        p['bc_setting'] = 'full'  # default boundary setting
        p['degree'] = 1  # polynomial degree
        #p['dim'] = 2  # default boundary setting
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
            #valbc = df.fem.Constant(self.mesh, ScalarType(0))
            #boundary_facets = df.mesh.locate_entities_boundary(self.mesh, self.p.dim -1, clamped_boundary)
            #displ_bcs.append(df.fem.dirichletbc(valbc, df.fem.locate_dofs_topological(V.sub(0), self.p.dim -1, boundary_facets), V.sub(0)))
            
        elif self.p.dim == 3:
            #displ_bcs.append(df.fem.DirichletBC(V, df.Constant((0, 0, 0)),  self.boundary_left()))
            displ_bcs.append(df.fem.dirichletbc(np.array([0, 0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(V, clamped_boundary), V))

        return displ_bcs

    def create_neumann_boundary(self):
        boundaries = [(1, lambda x: np.isclose(x[0], self.p.length))]

        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = df.mesh.locate_entities(self.mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = df.mesh.meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        
        
        #self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        #with df.io.XDMFFile(self.mesh.comm, "facet_tags.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.mesh)
        #    xdmf.write_meshtags(facet_tag)

        _ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
        return _ds