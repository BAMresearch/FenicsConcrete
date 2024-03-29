import dolfin as df

from fenics_concrete.material_problems.material_problem import MaterialProblem
from fenics_concrete.helpers import Parameters
from fenics_concrete import experimental_setups

# this is necessary, otherwise this warning will not stop
# https://fenics.readthedocs.io/projects/ffc/en/latest/_modules/ffc/quadrature/deprecation.html
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


class LinearElasticity(MaterialProblem):
    """Material definition for linear elasticity"""

    def __init__(self, experiment=None, parameters=None, pv_name='pv_output_linear_elasticity', vmapoutput = False):
        """Initializes the object by calling super().__init__

        Parameters
        ----------
            experiment : object, optional
                When no experiment is passed, the dummy experiment "MinimalCubeExperiment" is added
            parameters : dictionary, optional
                Dictionary with parameters. When none is provided, default values are used
            pv_name : string, optional
                Name of the paraview file, if paraview output is generated
        """
        # generate "dummy" experiment when none is passed
        if experiment is None:
            experiment = experimental_setups.MinimalCubeExperiment(parameters)

        super().__init__(experiment, parameters, pv_name, vmapoutput)

    def setup(self):
        default_p = Parameters()
        default_p['degree'] = 2  # polynomial degree

        # parameters for mechanics problem
        default_p['E'] = None  # Young's Modulus
        default_p['nu'] = None  # Poisson's Ratio
        default_p['mu'] = None
        default_p['lmbda'] = None

        self.p = default_p + self.p

        # expecting E and nu to compute mu and lambda, however you can directly supply mu and lambda
        # compute material parameters
        if self.p.mu is None or self.p.lmbda is None:
            assert self.p.E is not None and self.p.nu is not None
            self.p.mu = self.p.E / (2.0 * (1.0 + self.p.nu))
            self.p.lmbda = self.p.E * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

        # initialize possible paraview output
        self.pv_file = df.XDMFFile(self.pv_name + '.xdmf')
        self.pv_file.parameters["flush_output"] = True
        self.pv_file.parameters["functions_share_mesh"] = True

        # define function space ets.
        self.V = df.VectorFunctionSpace(self.experiment.mesh, "Lagrange", self.p.degree)  # 2 for quadratic elements

        self.residual = None  # initialize residual

        # Define variational problem
        self.u_trial = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        self.a = df.inner(self.sigma(self.u_trial), df.grad(v)) * df.dx

        if self.p.dim == 2:
            f = df.Constant((0, 0))
        elif self.p.dim == 3:
            f = df.Constant((0, 0, 0))
        else:
            raise Exception(f'wrong dimension {self.p.dim} for problem setup')

        self.L = df.inner(f, v) * df.dx

        # boundary conditions only after function space
        self.bcs = self.experiment.create_displ_bcs(self.V)

        # displacement field
        self.displacement = df.Function(self.V)

        # TODO better names!!!!
        self.visu_space_T = df.TensorFunctionSpace(self.experiment.mesh, "Lagrange", self.p.degree)

        if self.wrapper: self.wrapper.set_geometry(self.V, [self.a, self.L])

    # Stress computation for linear elastic problem
    def sigma(self, v):
        # v is the displacement field
        return 2.0 * self.p.mu * df.sym(df.grad(v)) + self.p.lmbda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))

    def solve(self, t = 1.0):
        if self.wrapper: self.wrapper.next_state()
        # solve
        df.solve(self.a == self.L, self.displacement, self.bcs)

        self.stress = self.sigma(self.displacement)

        # TODO make some switch in sensor definition to trigger this...
        self.compute_residual()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, self.wrapper, t)
        if self.wrapper: self.wrapper.write_state()

    def compute_residual(self):
        # compute reaction forces
        self.residual = df.action(self.a, self.displacement) - self.L

    def pv_plot(self, t=0):
        # paraview output

        # displacement plot
        u_plot = df.project(self.displacement, self.V)
        u_plot.rename("Displacement", "test string, what does this do??")  # TODO: what does the second string do?
        self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

        # stress plot
        sigma_plot = df.project(self.stress, self.visu_space_T)
        sigma_plot.rename("Stress", "test string, what does this do??")  # TODO: what does the second string do?
        self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
