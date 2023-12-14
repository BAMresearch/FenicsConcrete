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
        self.V_scalar = df.fem.FunctionSpace(self.mesh, ("Lagrange", self.p.degree, (self.mesh.geometry.dim-1,)))

        # boundary conditions only after function space
        self.bcs =[]
        #self.bcs.append(self.create_displ_bcs(self.V, self.p.dirichlet_bc[0], self.p.dirichlet_bc[1])) #2D case example fixing x boundary at location 0
        #self.bcs.append(self.create_displ_bcs(2, 0))  
        self.bcs.append(self.create_displ_bcs(1, self.p.dim_y, 0, 0, self.p.dim_x, 2, 0.1, 0.2))
        self.bcs.append(self.create_displ_bcs(1, self.p.dim_y, 0, 0, self.p.dim_x, 2, 0.8, 0.9))
        
        
    def create_displ_bcs(self, *args):
        # define displacement boundary      
        def clamped_boundary_2D(x):          # fenics will individually call this function for every node and will note the true or false value.
            if len(args) == 2:
                return np.isclose(x[args[0]], args[1])
            elif len(args) == 5:
                return np.logical_and(np.logical_and(np.isclose(x[args[0]], args[1]), np.logical_and(x[args[2]]>=args[3] , x[args[2]]<=args[4])), np.logical_and(x[args[5]]>=args[6] , x[args[5]]<=args[7]))
        
        def clamped_boundary_3D(x):
            if len(args) == 2:
                return np.isclose(x[args[0]], args[1])
            elif len(args) == 8:
                return np.logical_and(np.logical_and(np.isclose(x[args[0]], args[1]), np.logical_and(x[args[2]]>=args[3] , x[args[2]]<=args[4])), np.logical_and(x[args[5]]>=args[6] , x[args[5]]<=args[7]))
 
        if self.p.dim == 2:
            #displ_bcs.append(df.fem.DirichletBC(V, df.Constant((0, 0)), self.boundary_left()))
            return(df.fem.dirichletbc(np.array([0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(self.V, clamped_boundary_2D), self.V))

        elif self.p.dim == 3:
            return (df.fem.dirichletbc(np.array([0, 0, 0], dtype=ScalarType), df.fem.locate_dofs_geometrical(self.V, clamped_boundary_3D), self.V))
        
        else:
            print(f'wrong dimension {self.p.dim} for problem setup')
            exit()


    def identify_domain_boundaries(self):
        boundaries = [(1, lambda x: np.isclose(x[0], self.p.dim_x)), # x_upperlimit
            (2, lambda x: np.isclose(x[0], 0)), # x_lowlimit                              
            (3, lambda x: np.isclose(x[1], self.p.dim_y)),    # y_upperlimit
            (4, lambda x: np.isclose(x[1], 0))]   # y_lowlimit
        if self.p.dim == 3:
            boundaries.append((5, lambda x: np.isclose(x[2], self.p.dim_z)))   # z_upperlimit
            boundaries.append((6, lambda x: np.isclose(x[2], 0))) #z_lowlimit

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

        _ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
        return _ds
    
    def identify_domain_sub_boundaries(self, *args):
        if self.p.dim == 2:
            boundaries = [(1, lambda x: np.logical_and(np.isclose(x[args[0]], args[1]) , np.logical_and(x[args[2]]>=args[3] , x[args[2]]<=args[4])))] # right np.isclose(x[1], self.p.breadth) and 4500<x[0]<5000
        elif self.p.dim == 3:       
            boundaries = [(1, lambda x: np.logical_and(np.logical_and(np.isclose(x[args[0]], args[1]), np.logical_and(x[args[2]]>=args[3] , x[args[2]]<=args[4])), np.logical_and(x[args[5]]>=args[6] , x[args[5]]<=args[7])))]

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

        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        with df.io.XDMFFile(self.mesh.comm, "facet_tags.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_meshtags(facet_tag,self.mesh.geometry)
        
        _ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
        return _ds
