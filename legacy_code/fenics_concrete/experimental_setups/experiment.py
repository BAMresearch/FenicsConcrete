import dolfin as df
import numpy as np
from fenics_concrete.helpers import Parameters


class Experiment:
    """Parent class for experimental setups"""

    def __init__(self, parameters=None):
        """Initialises the parent object

        This is needs to be called by children
        Constant parameters are defined here
        """
        # setup of parameter field
        self.p = Parameters()
        # constants
        self.p['zero_C'] = 273.15  # to convert celsius to kelvin input to

        self.p = self.p + parameters

        self.setup()

    def setup(self):
        """Is called by init, must be defined by child"""
        raise NotImplementedError()

    # define some common boundary conditions
    def boundary_full(self):
        """Includes all nodes at the boundary"""

        def bc_full(x, on_boundary):
            return on_boundary

        return bc_full

    def boundary_empty(self):
        """Boundary condition without any nodes"""

        def bc_empty(x, on_boundary):
            return None

        return bc_empty

    def boundary_left(self):
        """Includes all nodes at the left of the mesh

        This is defined as the smallest x values in 2D and 3D (x[0])
        """
        # Left and right are defined as x (x[0]) in 2D and 3D
        # minimum vales aka "left" boundary
        left = np.amin(self.mesh.coordinates()[:, 0])

        def bc_left(x, on_boundary):
            return on_boundary and df.near(x[0], left)

        return bc_left

    def boundary_right(self):
        """Includes all nodes at the right of the mesh

        This is defined as the largest x values in 2D and 3D (x[0])
        """
        # Left and right are defined as x (x[0) in 2D and 3D
        # max vales aka "right" boundary
        right = np.amax(self.mesh.coordinates()[:, 0])

        def bc_right(x, on_boundary):
            return on_boundary and df.near(x[0], right)

        return bc_right

    def boundary_bottom(self, end=None):
        """Includes nodes at the bottom of the mesh

        This is defined as the smallest y values in 2D and the smallest z value 3D (x[2])

        Arguments
        ---------
        end : float, optional
            when defined, this excludes nodes with x values greater than end
            this has no practical function other than to create more complex test cases
        """

        if self.p.dim == 2:
            dir_id = 1
        elif self.p.dim == 3:
            dir_id = 2
        else:
            raise Exception('Dimension not defined')

        # minimum vales aka "bottom" boundary
        bottom = np.amin(self.mesh.coordinates()[:, dir_id])
        if end is None:
            end = np.amax(self.mesh.coordinates()[:, 0])

        def bc_bottom(x, on_boundary):
            return on_boundary and df.near(x[dir_id], bottom) and x[0] <= end

        return bc_bottom

    def boundary_top(self):
        """Includes all nodes at the top of the mesh

        This is defined as the largest y values (x[1]) in 2D and the largest z value 3D (x[2])
        """

        if self.p.dim == 2:
            dir_id = 1
        elif self.p.dim == 3:
            dir_id = 2
        else:
            raise Exception('Dimension not defined')

            # minimum vales aka "bottom" boundary
        top = np.amax(self.mesh.coordinates()[:, dir_id])

        def bc_top(x, on_boundary):
            return on_boundary and df.near(x[dir_id], top)

        return bc_top

    def boundary_front(self):
        """Includes all nodes at the front of the mesh

        This is only defined for the 3D case, as minimum y values (x[1])
        """
        # front and back are not defined in  2D and as y (x[1]) in 3D
        if self.p.dim == 2:
            bc = self.boundary_empty()

        elif self.p.dim == 3:
            # minimum vales aka "front" boundary
            front = np.amin(self.mesh.coordinates()[:, 1])

            def bc_front(x, on_boundary):
                return on_boundary and df.near(x[1], front)

            bc = bc_front
        else:
            raise Exception('Dimension not defined')

        return bc

    def boundary_back(self):
        """Includes all nodes at the back of the mesh

        This is only defined for the 3D case, as max y values (x[1])
        """

        if self.p.dim == 2:
            bc = self.boundary_empty()

        elif self.p.dim == 3:
            # max vales aka "back" boundary
            back = np.amax(self.mesh.coordinates()[:, 1])

            def bc_back(x, on_boundary):
                return on_boundary and df.near(x[1], back)

            bc = bc_back
        else:
            raise Exception('Dimension not defined')

        return bc
