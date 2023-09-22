# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, SpikeAndSlab, Bernoulli, RegularisedHorseShoe, HalfCauchy, InvGamma, HorseShoe
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (problem solving)
from probeye.inference.scipy.solver import MaxLikelihoodSolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.dynesty.solver import DynestySolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

# ground truth

a_true = 8
b_true = 10

# settings for data generation
n_tests = 50
seed = 1
mean_noise = 0.0
std_noise = 1

# generate the data
np.random.seed(seed)
x_test = np.linspace(-4.0, 4.0, n_tests)
y_true = a_true * x_test + b_true #a_true * x_test**2 #+ b_true * x_test + c_true #a_true * x_test**2 + 
y_test = y_true + np.random.normal(loc=mean_noise, scale=std_noise, size=n_tests)

class LinearModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["a", "b"] #, "c"
        self.input_sensors = Sensor("x")
        self.output_sensors = Sensor("y", std_model="sigma")

    def response(self, inp: dict) -> dict:
        x = inp["x"]
        a = inp["a"]
        b = inp["b"]
        return {"y":  a * x + b} #a * x ** 2 + 
    

# initialize the problem (the print_header=False is only set to avoid the printout of
# the probeye header which is not helpful here)
problem = InverseProblem("Linear regression with Gaussian noise", print_header=False)

# add the problem's parameters
problem.add_parameter(
    "lmbda_a",
    #tex="$\lambda_a$",
    #info="Lambda of Slope of the graph",
    #domain="[0, +oo)",
    prior=HalfCauchy(loc=0, scale=1),
)

problem.add_parameter(
    "c_sqr_a",
    #tex="$\lambda_a$",
    #info="Lambda of Slope of the graph",
    domain="[0, +oo)",
    prior=InvGamma(shape=2, scale=8),
)

problem.add_parameter(
    "tau_a",
    #tex="$\lambda_a$",
    #info="Lambda of Slope of the graph",
    #domain="[0, +oo)",
    prior=HalfCauchy(loc=0, scale=1),
)

problem.add_parameter(
    "a",
    tex="$a$",
    #info="Slope of the graph",
    prior=RegularisedHorseShoe(mean=0, lmbda="lmbda_a", c_sqr="c_sqr_a", tau="tau_a"),
)

""" problem.add_parameter(
    "a",
    tex="$a$",
    #info="Slope of the graph",
    prior=HorseShoe(mean=0, lmbda="lmbda_a", tau="tau_a"),
) """

problem.add_parameter(
    "b",
    #tex="$\lambda_b$",
    #info="Lambda of Slope of the graph",
    prior=Uniform(low=8, high=12),
)
problem.add_parameter(
    "sigma",
    domain="(0, +oo)",
    tex=r"$\sigma$",
    info="Standard deviation, of zero-mean Gaussian noise model",
    #prior=Uniform(low=0.0, high=1.5),
    value = 1.0
)


# experimental data
problem.add_experiment(
    name="TestSeries_1",
    sensor_data={"x": x_test, "y": y_test},
)

# forward model
problem.add_forward_model(LinearModel("LinearModel"), experiments="TestSeries_1")

# likelihood model
problem.add_likelihood_model(
    GaussianLikelihoodModel(experiment_name="TestSeries_1", model_error="additive")
)

# print problem summary
problem.info(print_header=True)


# this is for using the emcee-solver (MCMC sampling)
emcee_solver = EmceeSolver(problem, show_progress=True)
inference_data = emcee_solver.run(n_steps=2000, n_initial_steps=1, n_walkers=20)
posterior = emcee_solver.raw_results.get_chain()


# this is optional, since in most cases we don't know the ground truth
#true_values = {"a": a_true, "b": b_true, "lmbda_a": 0, "tau_a": 0}
true_values = {"a": a_true, "b": b_true, "lmbda_a": 0, "c_sqr_a":0, "tau_a": 0}
#true_values = {"a": 0, "b": b_true, "lmbda_a": 0, "sigma": std_noise}

# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    emcee_solver.problem,
    true_values=true_values,
    focus_on_posterior=True,
    show_legends=True,
    title="Sampling results from emcee-Solver (pair plot)",
)

# trace plots are used to check for "healthy" sampling
trace_plot_array = create_trace_plot(
    inference_data,
    emcee_solver.problem,
    title="Sampling results from emcee-Solver (trace plot)",
)