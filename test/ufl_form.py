from ufl import *

cell = quadrilateral # an object of type cell. Other options: triangle, interval, hexahedron, tetrahedron
print(cell.num_vertices(), cell.geometric_dimension(), cell.topological_dimension())

# Create a cell of type triangle
cell2 = Cell("triangle", 2)
print(cell2.num_vertices(), cell2.geometric_dimension(), cell2.topological_dimension())


# Basic elements
element1 = FiniteElement("Lagrange", cell, 1)
element2 = FiniteElement("DG", cell, 1)

#Mixed Element : Must be based on the same cell type
element3 = element1 * element2
print(element3)

########################################################################################################################
#Arguments: Trial and Test function
element = VectorElement("CG", triangle, 2)
# vq = Argument(element) #Doesn't work
vq = TrialFunction(element)
print(vq)

f=Coefficient(element)
#f1 = Constant(element) #Doesn't work
print(f)

##########################################################################################################################
#Basic Datatypes
e1 = FiniteElement("CG", cell, 1)
v = TestFunction(e1)
x  = SpatialCoordinate(cell)
L = sin(x[1])*v*dx
#print(x.ufl_shape)

# Geometric dimension
d = cell.geometric_dimension()

# d x d identiy matrix
I = Identity(d)

# Kronecker delta
delta_ij = I[i,j]

############################################################################################################################
#Form Transformations
