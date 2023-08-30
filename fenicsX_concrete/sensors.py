import dolfinx as df
import numpy as np


class Sensors(dict):
    """
    Dict that also allows to access the parameter p["parameter"] via the matching attribute p.parameter
    to make access shorter

    When to sensors with the same name are defined, the next one gets a number added to the name
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def __setitem__(self, initial_key, value):
        # check if key exists, if so, add a number to the name
        i = 2
        key = initial_key
        if key in self:
            while key in self:
                key = initial_key + str(i)
                i += 1

        super().__setitem__(key, value)


# sensor template
class Sensor:
    """Template for a sensor object"""

    def measure(self, problem, t):
        """Needs to be implemented in child, depending on the sensor"""
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__

    def data_max(self, value):
        if value > self.max:
            self.max = value


class DisplacementSensor(Sensor):
    """A sensor that measure displacement at a specific point"""

    def __init__(self, where, alphabetical_position):
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.alphabetical_position = alphabetical_position
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get displacements
        #self.data.append(problem.displacement(self.where))

        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []
        points_on_proc = []

        # Find cells whose bounding-box collide with the point
        cell_candidates = df.geometry.compute_collisions(bb_tree, self.where)

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, self.where)
        for i, point in enumerate(self.where):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        self.data.append(problem.displacement.eval(points_on_proc, cells))
        self.time.append(t)


class TemperatureSensor(Sensor):
    """A sensor that measure temperature at a specific point in celsius"""

    def __init__(self, where):
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        T = problem.temperature(self.where) - problem.p.zero_C
        self.data.append(T)
        self.time.append(t)

class StressSensor(Sensor):
    """A sensor that measure the stress tensor in at a point"""

    def __init__(self, where):
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get stress
        stress = df.project(problem.stress, problem.visu_space_T, form_compiler_parameters={'quadrature_degree': problem.p.degree})
        self.data.append(stress(self.where))
        self.time.append(t)

class StrainSensor(Sensor):
    """A sensor that measure the strain tensor in at a point"""

    def __init__(self, where):
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get displacements
        #self.data.append(problem.displacement(self.where))

        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []
        points_on_proc = []

        # Find cells whose bounding-box collide with the point
        cell_candidates = df.geometry.compute_collisions(bb_tree, self.where)

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, self.where)
        for i, point in enumerate(self.where):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        self.data.append(problem.strain.eval(points_on_proc, cells))
        self.time.append(t)

class DisplacementDoubleDerivativeSensor(Sensor):
    """A sensor that measure the strain tensor in at a point"""

    def __init__(self, where):
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get displacements
        #self.data.append(problem.displacement(self.where))

        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []
        points_on_proc = []

        # Find cells whose bounding-box collide with the point
        cell_candidates = df.geometry.compute_collisions(bb_tree, self.where)

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, self.where)
        for i, point in enumerate(self.where):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        self.data.append(problem.displacement_double_derivative.eval(points_on_proc, cells))
        self.time.append(t)


class ReactionForceSensor(Sensor):
    """A sensor that measure the reaction force at the bottom perpendicular to the surface"""

    def __init__(self):
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # boundary condition
        # bottom_surface = problem.experiment.boundary_bottom()

        #self.x_force = np.sum(self.residual_numeric.array[self.experiment.bc_x_dof])
        #self.y_force = np.sum(self.residual_numeric.array[self.experiment.bc_y_dof])  

        force_x_vector = problem.residual_numeric.array[problem.experiment.bc_x_dof]
        force_y_vector = problem.residual_numeric.array[problem.experiment.bc_y_dof]
        self.x_force = np.sum(problem.residual_numeric.array[problem.experiment.bc_x_dof])
        self.y_force = np.sum(problem.residual_numeric.array[problem.experiment.bc_y_dof])  

        self.data.append(np.array((self.x_force,self.y_force)))
        self.time.append(t)
                
#    def measure(self, problem, t=1.0):
#        """
#        Arguments:
#            problem : FEM problem object
#            t : float, optional
#                time of measurement for time dependent problems
#        """
#        # get strain
#        strain = df.project(problem.strain, problem.visu_space_T, form_compiler_parameters={'quadrature_degree': problem.p.degree})
#        self.data.append(strain(self.where))
#        self.time.append(t)