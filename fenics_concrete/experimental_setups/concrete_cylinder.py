from fenics_concrete.experimental_setups.experiment import Experiment
from fenics_concrete.helpers import Parameters
import dolfin as df
import numpy as np
import gmsh
import os
import meshio


def generate_cylinder_mesh(radius,height,mesh_density):
    '''Uses gmsh to generate a cylinder mesh for fenics

    Paramters
    ---------
    radius : float
        radius of the cylinder
    height : float
        height of the cylinder
    mesh_density : float
        defines the size of the elements
        defines the minimum number of element edges in the height of the cylinder

    Returns
    -------
    mesh : dolfin.cpp.mesh.Mesh object
        cylinder mesh for dolfin
    '''

    # file names and location
    folder_name = 'mesh'
    file_name = 'cylinder'
    msh_file = '{}/{}.msh'.format(folder_name,file_name)
    xdmf_file = '{}/{}.xdmf'.format(folder_name,file_name)

    # start gmsh
    gmsh.initialize()
    gmsh.option.set_number('General.Verbosity',3) # only print warnings etc
    gmsh.model.add('cylinder_mesh')  # give the model a name

    # generate cylinder geometry with origin in (0,0,0)
    # syntax: add_cylinder(x,y,z,dx,dy,dz,radius,angle in radian)
    membrane = gmsh.model.occ.add_cylinder(0,0,0,0,0,height,radius,angle=2*np.pi)
    gmsh.model.occ.synchronize()
    dim = 3
    # only physical groups get exported
    # syntax: add_physical_group(dim , list of 3d objects, tag)
    gmsh.model.add_physical_group(dim, [membrane], 1)

    # meshing
    characteristic_length = height/mesh_density
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",characteristic_length)
    # setting for minimal length, arbitrarily chosen as half the max value
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",characteristic_length/2)
    gmsh.model.mesh.generate(dim)

    # save it to disk as msh in folder
    if not os.path.exists(folder_name):  # creat mesh folder if it does nto exists
        os.mkdir(folder_name)

    # write file
    gmsh.write(msh_file)
    gmsh.finalize()

    # convert msh to xdmf
    meshio_mesh = meshio.read(msh_file)
    meshio.write(xdmf_file, meshio_mesh)

    # read xdmf as dolfin mesh
    mesh = df.Mesh()
    with df.XDMFFile(xdmf_file) as mesh_file:
        mesh_file.read(mesh)

    return mesh


class ConcreteCylinderExperiment(Experiment):
    """A cylinder mesh for a uni-axial displacement load"""

    def __init__(self, parameters=None):
        """ initializes the object

        Standard parameters are set
        setup function called

        Parameters
        ----------
        parameters : dictionary, optional
            dictionary with parameters that can override the default values
        """
        # initialize a set of default parameters
        p = Parameters()
        # boundary setting
        p['bc_setting'] = 'free'  # boundary setting, two options available: fixed and free
                                  # fixed: constrained at top and bottom in transversal to loading
                                  # free: no confinement perpendicular to loading surface
        # mesh information
        p['dim'] = 3  # dimension of problem, 2D or 3D
                      # 2D version of the cylinder is a rectangle with plane strain assumption
        p['mesh_density'] = 4  # in 3D: number of faces on the side when generating a polyhedral approximation
                               # in 2D: number of elements in each direction
        p['radius'] = 75   # radius of cylinder to approximate in mm
        p['height'] = 100  # height of cylinder in mm

        p = p + parameters
        super().__init__(p)

        # initialize variable top_displacement
        self.top_displacement = df.Constant(0.0)  # applied via fkt: apply_displ_load(...)

    def setup(self):
        """Generates the mesh based on parameters

        This function is called during __init__
        """

        if self.p.dim == 2:
            # build a rectangular mesh to approximate a 2D cylinder
            self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.p.radius*2, self.p.height),
                                         self.p.mesh_density, self.p.mesh_density, diagonal='right')
        elif self.p.dim == 3:
            # generates a 3D cylinder mesh based on radius and height
            # to reduce approximation errors due to the linear tetrahedron mesh, the mesh radius is iteratively changed
            # until the bottom surface area matches that of a circle with the initially defined radius
            def create_cylinder_mesh(radius, parameters):
                # generate cylinder mesh using gmsh
                mesh = generate_cylinder_mesh(radius, parameters.height, parameters.mesh_density)

                # compute bottom surface area
                class BottomSurface(df.SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary and df.near(x[2], 0.0)

                bottom_surface = BottomSurface()
                boundaries = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
                boundaries.set_all(0)
                bottom_surface.mark(boundaries, 1)
                ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
                bottom_area = df.assemble(1 * ds(1))

                return bottom_area, mesh

            # create a discretized cylinder mesh with the same cross-sectional area as the round cylinder
            target_area = np.pi*self.p.radius**2
            effective_radius = self.p.radius
            mesh_area = 0
            area_error = 1e-6
            #
            #iteratively improve the radius of the mesh till the bottom area matches the target
            while abs(target_area - mesh_area) > target_area*area_error:
                # generate mesh
                self.p['mesh_radius'] = effective_radius  # not required, but maybe interesting as metadata
                mesh_area, self.mesh = create_cylinder_mesh(effective_radius, self.p)
                # new guess
                effective_radius = np.sqrt(target_area/mesh_area)*effective_radius

        else:
            raise Exception(f'wrong dimension {self.p.dim} for problem setup')

    def create_displ_bcs(self, V):
        """Defines the displacement boundary conditions

        Parameters
        ----------
            V : FunctionSpace
                Function space of the structure

        Returns
        -------
            displ_bc : list
                A list of DirichletBC objects, defining the boundary conditions
        """

        # define displacement boundary
        displ_bcs = []

        def boundary_node_2D(point):
            def bc_node(x, on_boundary):
                return df.near(x[0], point[0]) and df.near(x[1], point[1])
            return bc_node

        def boundary_node_3D(point):
            def bc_node(x, on_boundary):
                return df.near(x[0], point[0]) and df.near(x[1], point[1]) and df.near(x[2], point[2])
            return bc_node

        if self.p.bc_setting == 'fixed':
            if self.p.dim == 2:
                displ_bcs.append(df.DirichletBC(V.sub(1), self.top_displacement, self.boundary_top()))  # displacement
                displ_bcs.append(df.DirichletBC(V.sub(0), 0, self.boundary_top()))
                displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), self.boundary_bottom()))

            elif self.p.dim == 3:
                displ_bcs.append(df.DirichletBC(V.sub(2), self.top_displacement, self.boundary_top()))  # displacement
                displ_bcs.append(df.DirichletBC(V.sub(0), 0, self.boundary_top()))
                displ_bcs.append(df.DirichletBC(V.sub(1), 0, self.boundary_top()))
                displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0, 0)),  self.boundary_bottom()))

        elif self.p.bc_setting == 'free':
            if self.p.dim == 2:
                displ_bcs.append(df.DirichletBC(V.sub(1), self.top_displacement, self.boundary_top()))  # displacement
                displ_bcs.append(df.DirichletBC(V.sub(1), 0.0, self.boundary_bottom()))
                displ_bcs.append(df.DirichletBC(V.sub(0), 0.0, boundary_node_2D(df.Point(0, 0)), method="pointwise"))

            elif self.p.dim == 3:
                # getting nodes at the bottom of the mesh to apply correct boundary condition to arbitrary cylinder mesh
                mesh_points = self.mesh.coordinates()  # list of all nodal coordinates
                bottom_points = mesh_points[(mesh_points[:, 2] == 0.0)]  # copying the bottom nodes, z coord = 0.0

                # sorting by x coordinate
                x_min_boundary_point = bottom_points[bottom_points[:, 0].argsort(kind='mergesort')][0]
                x_max_boundary_point = bottom_points[bottom_points[:, 0].argsort(kind='mergesort')][-1]
                # sorting by y coordinate
                y_boundary_point = bottom_points[bottom_points[:, 1].argsort(kind='mergesort')][0]

                displ_bcs.append(df.DirichletBC(V.sub(2), self.top_displacement, self.boundary_top()))  # displacement
                displ_bcs.append(df.DirichletBC(V.sub(2), 0.0, self.boundary_bottom()))
                displ_bcs.append(df.DirichletBC(V.sub(1), 0.0, boundary_node_3D(x_min_boundary_point), method="pointwise"))
                displ_bcs.append(df.DirichletBC(V.sub(1), 0.0, boundary_node_3D(x_max_boundary_point), method="pointwise"))
                displ_bcs.append(df.DirichletBC(V.sub(0), 0.0, boundary_node_3D(y_boundary_point), method="pointwise"))
            else:
                raise Exception(f'dim setting: {self.p.dim}, not implemented for cylinder bc setup: free')

        else:
            raise Exception(f'Wrong boundary setting: {self.p.bc_setting}, for cylinder setup')

        return displ_bcs

    def apply_displ_load(self, top_displacement):
        """Updates the applied displacement load

        Parameters
        ----------
        top_displacement : float
            Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression
        """

        self.top_displacement.assign(df.Constant(top_displacement))

