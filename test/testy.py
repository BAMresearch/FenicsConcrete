
"""
              Linear model in time and space with three different sensors
----------------------------------------------------------------------------------------
                       ---> Additive model prediction error <---
----------------------------------------------------------------------------------------
The model equation is y(x,t) = a * x + b * t + c with a, b, c being the model parameters
while x and t represent position and time respectively. From the three model parameters
a and b are latent ones while c is a constant. Measurements are made at three different
positions (x-values) each of which is associated with an own zero-mean, uncorrelated
normal model error with the standard deviations to infer. This results in five latent
parameters (parameters to be inferred). The problem is approached with a maximum
likelihood estimation.
"""


import os

# third party imports
import numpy as np
from probeye.inference.emcee.solver import EmceeSolver
# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (knowledge graph)
from probeye.ontology.knowledge_graph_export import export_knowledge_graph

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


n_steps: int = 200,
n_initial_steps: int = 100,
n_walkers: int = 20,
plot: bool = False,
show_progress: bool = False,
write_to_graph: bool = True,
run_scipy: bool = True,
run_emcee: bool = False,  # intentionally False for faster test-runs
run_dynesty: bool = False,  # intentionally False for faster test-runs

    # ============================================================================ #
    #                              Set numeric values                              #
    # ============================================================================ #
# 'true' value of A, and its normal prior parameters
a_true = 1.3
mean_a = 1.0
std_a = 1.0
# 'true' value of B, and its normal prior parameters
b_true = -1.0
mean_b = -2.0
std_b = 1.5
# 'true' value of sd_S1, and its uniform prior parameters
sd_s1_true = 0.2
low_s1 = 0.0
high_s1 = 0.7
# 'true' value of sd_S2, and its uniform prior parameters
sd_s2_true = 0.4
low_s2 = 0.0
high_s2 = 0.7
# 'true' value of sd_S3, and its uniform prior parameters
sd_s3_true = 0.6
low_s3 = 0.0
high_s3 = 0.7
# define sensor positions
pos_s1 = 0.2
pos_s2 = 0.5
pos_s3 = 1.0
# define global constants
sigma_m = 0.1
sigma_n = 0.2
c = 0.5
# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #
# initialize the inverse problem with a useful name
problem = InverseProblem("Linear model with three sensors")
# add all parameters to the problem
problem.add_parameter(
    name="a",
    prior=Normal(mean=mean_a, std=std_a),
    info="Slope of the graph in x",
    tex="$A$",
)
problem.add_parameter(
    name="b",
    prior=Normal(mean=mean_b, std=std_b),
    info="Slope of the graph in t",
    tex="$B$",
)
problem.add_parameter(
    name="sigma_1",
    domain="(0, +oo)",
    prior=Uniform(low=low_s1, high=high_s1),
    info="Standard deviation, of zero-mean additive model error for S1",
    tex=r"$\sigma_1$",
)
problem.add_parameter(
    name="sigma_2",
    domain="(0, +oo)",
    prior=Uniform(low=low_s2, high=high_s2),
    info="Standard deviation of zero-mean additive model error for S2",
    tex=r"$\sigma_2$",
)
problem.add_parameter(
    name="sigma_3",
    domain="(0, +oo)",
    prior=Uniform(low=low_s3, high=high_s3),
    info="Standard deviation of zero-mean additive model error S3",
    tex=r"$\sigma_3$",
)
problem.add_parameter(
    name="sigma_m",
    prior=Uniform(low=0.07, high=0.13),
    #value=sigma_m,
    info="Standard deviation of zero-mean additive measurement error",
)
problem.add_parameter(
    name="sigma_n",
    prior=Uniform(low=0.08, high=0.12),
    #value=sigma_n,
    info="Standard deviation of zero-mean additive measurement error",
)
problem.add_parameter(
    name="c",
    value=c,
    info="Known model constant of forward model",
)
# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #
# add the experimental data
np.random.seed(1)
def generate_data(n_time_steps, idx=None):
    # true values
    time_steps = np.linspace(0, 1, n_time_steps)
    sensor_data = {
        "time": time_steps,
        "y1": a_true * pos_s1 + b_true * time_steps + c,
        "y2": a_true * pos_s2 + b_true * time_steps + c,
        "y3": a_true * pos_s3 + b_true * time_steps + c,
    }
    # add noise
    sensor_data["y1"] += np.random.normal(
        0.0, np.sqrt(sd_s1_true**2 + sigma_m**2), size=n_time_steps
    )
    sensor_data["y2"] += np.random.normal(
        0.0, np.sqrt(sd_s2_true**2 + sigma_m**2), size=n_time_steps
    )
    sensor_data["y3"] += np.random.normal(
        0.0, np.sqrt(sd_s3_true**2 + sigma_m**2), size=n_time_steps
    )
    # add experiment to problem
    problem.add_experiment(name=f"TestSeries_{idx}", sensor_data=sensor_data)
# generate the data for fitting
for i, n_t in enumerate([101, 51]):
    generate_data(n_t, idx=i + 1)
# ============================================================================ #
#                           Define the Forward Model                           #
# ============================================================================ #
class LinearModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["a", "b", {"c": "const"}]
        self.input_sensors = Sensor("time")
        self.output_sensors = [
            Sensor("y1", x=pos_s1, std_model="sigma_1"),
            Sensor("y2", x=pos_s2, std_model="sigma_2"),
            Sensor("y3", x=pos_s3, std_model="sigma_3"),
        ]
    def response(self, inp: dict) -> dict:
        t = inp["time"]
        a = inp["a"]
        b = inp["b"]
        const = inp["const"]
        response = dict()
        for osensor in self.output_sensors:
            response[osensor.name] = a * osensor.x + b * t + const
        return response
# add the forward model to the problem
linear_model = LinearModel("LinearModel")
problem.add_forward_model(
    linear_model, experiments=["TestSeries_1", "TestSeries_2"]
)
# ============================================================================ #
#                           Add likelihood model(s)                            #
# ============================================================================ #
# add the likelihood models to the problem
problem.add_likelihood_model(
GaussianLikelihoodModel(
    experiment_name="TestSeries_1",
    model_error="additive",
    measurement_error="sigma_m",
))
problem.add_likelihood_model(
    GaussianLikelihoodModel(
        experiment_name="TestSeries_2",
        model_error="additive",
        measurement_error="sigma_n",
    )
)
# give problem overview
problem.info()
emcee_solver = EmceeSolver(problem)
inference_data = emcee_solver.run(n_steps=200, n_initial_steps=100,n_walkers=20)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot
# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    emcee_solver.problem,
    #true_values=true_values,
    focus_on_posterior=True,
    show_legends=True,
    title="Sampling results from emcee-Solver (pair plot)",
)

trace_plot_array = create_trace_plot(
    inference_data,
    emcee_solver.problem,
    title="Sampling results from emcee-Solver (trace plot)",
)