
""" 

import dolfin as df

mesh = df.UnitIntervalMesh(5)



E_num = 1.
elem_u = df.FiniteElement('CG', mesh.ufl_cell(), 1)
V = df.FunctionSpace(mesh, elem_u)
u = df.Function(V)
u.vector()[:] = 1.
E = df.Constant(E_num)
lhs = E * u * df.dx



print(f"first value = {df.assemble(lhs)}")
E.assign(2.)
print(f"2nd value = {df.assemble(lhs)}")
E.assign(20.)
print(f"2nd value = {df.assemble(lhs)}") """

import dolfinx as df
from mpi4py import MPI
from petsc4py import PETSc

mesh1 = df.mesh.create_rectangle(comm=MPI.COMM_WORLD, points=((0.0, 0.0), (20, 0.2)), n=(20, 10), cell_type=df.mesh.CellType.quadrilateral)
#print(help(df.fem.Constant))
E = df.fem.Constant(mesh1, 2.)

""" def func1():
    print(E.value*2)
func1()
print(E.value)
E.value = 3
func1()
print(E.value) """


A = 2*E.value
print(A)
E.value = 3.

print(A)


