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
        p = p + parameters
        super().__init__(p)

    def setup(self, bc='full'):
        self.bc = bc  # different boundary settings

        # elements per spatial direction
        if self.p.dim == 2:
            #self.mesh = df.UnitSquareMesh(n, n, self.p.mesh_setting)
            self.mesh = df.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (self.p.dim_x, self.p.dim_y)), n=(self.p.num_elements_x, self.p.num_elements_y),
                            cell_type=df.mesh.CellType.quadrilateral)
        elif self.p.dim == 3:
            self.mesh = df.mesh.create_box(comm=MPI.COMM_WORLD, points=[np.array([0, 0, 0]), np.array([self.p.dim_x, self.p.dim_y, self.p.dim_z])],
                         n=[self.p.num_elements_x, self.p.num_elements_y, self.p.num_elements_z], cell_type=df.mesh.CellType.hexahedron)         
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()

        # define function space ets.
        self.V = df.fem.functionspace(self.mesh, ("Lagrange", self.p.degree, (self.mesh.geometry.dim,))) # 2 for quadratic elements
        #self.V_scalar = df.fem.FunctionSpace(self.mesh, ("Lagrange", self.p.degree, (self.mesh.geometry.dim-1,)))

        # Dirichlet boundary
        dirichlet_bdy_sub1 = self.boundary_locator([2, 0])
        #dirichlet_bdy_sub1 = self.boundary_locator([1, self.p.dim_y, 0, 0, self.p.dim_x, 2, 0.1, 0.2])
        #dirichlet_bdy_sub2 = self.boundary_locator([1, self.p.dim_y, 0, 0, self.p.dim_x, 2, 0.8, 0.9+1e-5])

        self.bcs =[]
        self.bcs.append(self.create_displ_bc(dirichlet_bdy_sub1))
        #self.bcs.append(self.create_displ_bc(dirichlet_bdy_sub2))

        dirichlet_bdy = [(1, dirichlet_bdy_sub1),]

        #dirichlet_bdy = [(1, dirichlet_bdy_sub1),
        #                 (2, dirichlet_bdy_sub2)]
        
        self.create_facet_tag(dirichlet_bdy, "facet_tags_dirichlet.xdmf")

        # Neumann boundary
        
        neumann_bdy_sub1 = self.boundary_locator([1, 0, 0, self.p.lower_limit_x, self.p.upper_limit_x, 2, self.p.lower_limit_z, self.p.upper_limit_z])
        
        neumann_bdy = [(1, neumann_bdy_sub1)]

        self.ds = self.create_facet_tag(neumann_bdy, "facet_tags_neumann.xdmf", True)


    def boundary_locator(self, bdy_def):
        if len(bdy_def) == 2:
            return lambda x : np.isclose(x[bdy_def[0]], bdy_def[1])
        
        elif len(bdy_def) == 5:
            return lambda x : np.logical_and(np.isclose(x[bdy_def[0]], bdy_def[1]) , np.logical_and(x[bdy_def[2]]>=bdy_def[3] , x[bdy_def[2]]<=bdy_def[4]))
        
        elif len(bdy_def) == 8:
            return lambda x : np.logical_and(np.logical_and(np.isclose(x[bdy_def[0]], bdy_def[1]), 
                                            np.logical_and(x[bdy_def[2]]>=bdy_def[3] , x[bdy_def[2]]<=bdy_def[4])), 
                                            np.logical_and(x[bdy_def[5]]>=bdy_def[6] , x[bdy_def[5]]<=bdy_def[7]))

    def create_displ_bc(self, boundary_locator):
        if self.p.dim == 2:
            #displ_bcs.append(df.fem.DirichletBC(V, df.Constant((0, 0)), self.boundary_left()))
            return(df.fem.dirichletbc(np.array([0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(self.V, boundary_locator), self.V))

        elif self.p.dim == 3:
            return (df.fem.dirichletbc(np.array([0, 0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(self.V, boundary_locator), self.V))
        
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()
    
    def create_facet_tag(self, boundary, file_name, ufl_bdy=False):
        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1
        for (marker, locator) in boundary:
            facets = df.mesh.locate_entities(self.mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = df.mesh.meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        with df.io.XDMFFile(self.mesh.comm, file_name, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_meshtags(facet_tag,self.mesh.geometry)
        
        if ufl_bdy == True:
            _ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
            return _ds
