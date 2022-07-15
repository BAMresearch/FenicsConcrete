from fenics_concrete.experimental_setups.experiment import Experiment
from fenics_concrete.helpers import Parameters
import dolfin as df

# layered experiment set-up in the moment only 2D
# one mesh for all layers -> activate elements by pseudo-density

class MultipleLayers2DExperiment(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        p['dim'] = 2  # default boundary setting
        p['stress_case'] = 'plane_strain'
        p['mesh_density'] = 2  # number of elements in layer height -> other will be set
        p['layer_width'] = 4/100 #m can be used as crosssection or length!
        p['layer_height'] = 1/100 #m
        p['layer_number'] = 10 # number of layers
        p = p + parameters
        super().__init__(p)

        # initialize variable top_displacement
        self.top_displacement = df.Constant(0.0)

    def setup(self):
        # elements per spacial direction
        n_height_layer = int(self.p['mesh_density'])
        n_width = int(n_height_layer / self.p.layer_height * self.p.layer_width)

        if self.p.dim == 2:
             self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.p.layer_width, self.p.layer_number*self.p.layer_height),
                                          n_width,
                                          int(self.p.layer_number)*n_height_layer, diagonal='right')
        else:
            raise ValueError('Dimension has to be 2!! for that experiment')

    def create_displ_bcs(self, V):
        # define displacement boundary, fixed at bottom
        displ_bcs = []

        if self.p.dim == 2:
            displ_bcs.append(df.DirichletBC(V, df.Constant((0, 0)), self.boundary_bottom()))
        else:
            raise ValueError('Dimension has to be 2!! for that experiment')

        return displ_bcs