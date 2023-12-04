from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner)

from mpi4py import MPI


import numpy as np
import pyvista

pyvista.start_xvfb()

mesh = create_unit_square(MPI.COMM_WORLD, 1, 1)
Q = FunctionSpace(mesh, ("DG", 1))
print(Q.tabulate_dof_coordinates())