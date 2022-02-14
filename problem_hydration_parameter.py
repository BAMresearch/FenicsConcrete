from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

#import probeye
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
# local imports (problem definition)
# from probeye.definition.inference_problem import InferenceProblem
# from probeye.definition.forward_model import ForwardModelBase
# from probeye.definition.sensor import Sensor
# from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


# ============================================================================ #
#                           Define the Forward Model                           #
# ============================================================================ #

class LinearModel(ForwardModelBase):
    def definition(self):
        self.parameters = [{"a": "m"}, "b"]
        self.input_sensors = Sensor("x")
        self.output_sensors = Sensor("y")

    def response(self, inp: dict) -> dict:
        # this method *must* be provided by the user
        x = inp["x"]
        m = inp["m"]
        b = inp["b"]
        return {"y": m * x + b}

    def jacobian(self, inp: dict) -> dict:
        # this method *can* be provided by the user; if not provided the
        # jacobian will be approximated by finite differences
        x = inp["x"]  # vector
        one = np.ones((len(x), 1))
        # partial derivatives must only be stated for the model parameters;
        # all other input must be flagged by None; note: partial derivatives
        # must be given as column vectors
        return {"y": {"x": None, "m": x.reshape(-1, 1), "b": one}}

class HydrationHeatModel(ForwardModelBase):
    def definition(self):
        self.parameters = ['B1', 'B2','eta']

        self.input_sensors = Sensor("T")
        self.input_sensors = Sensor("dt")
        self.input_sensors = Sensor("time_max")
        self.input_sensors = Sensor("E_act")
        self.input_sensors = Sensor("Q_pot")
        self.input_sensors = Sensor("T_ref")

        self.output_sensors = Sensor("alpha_max")
        self.output_sensors = Sensor("time")

    def response(self, inp: dict) -> dict:
        # this method *must* be provided by the user
        T = inp["T"]
        dt = inp["dt"]
        time_max = inp["time_max"]
        parameter = {}
        parameter['B1'] = inp["B1"]
        parameter['B2'] = inp["B2"]
        parameter['eta'] = inp["eta"]
        parameter['alpha_max'] = inp["alpha_max"]
        parameter['E_act'] = inp["E_act"]
        parameter['T_ref'] = inp["T_ref"]
        parameter['Q_pot'] = inp["Q_pot"]

        time_list, heat_list, doh_list = hydration_fkt(T, time_max, dt, parameter)
        return {"time": time_list, 'heat': heat_list}



#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------

# read data
T = 15
time_data = []
heat_data = []
with open(f'test_hydration_data_T{T}.dat') as f:
    for line in f:
        if line[0] != '#':
            vals = line.split()
            time_data.append(float(vals[0]))
            heat_data.append(float(vals[0]))


# initiate material problem
material_problem = concrete_problem.ConcreteThermoMechanical()
# get the respective function
hydration_fkt = material_problem.get_heat_of_hydration_ftk()

# set required parameter
parameter = concrete_experiment.Parameters() # using the current default values

parameter['B1'] = 2.916E-4  # in 1/s
parameter['B2'] = 0.0024229  # -
parameter['eta'] = 5.554  # something about diffusion
parameter['alpha_max'] = 0.875  # also possible to approximate based on equation with w/c
parameter['E_act'] = 47002   # activation energy in Jmol^-1
parameter['T_ref'] = 25  # reference temperature in degree celsius
parameter['igc'] = 8.3145  # ideal gas constant in [J/K/mol], CONSTANT!!!
parameter['zero_C'] = 273.15  # in Kelvin, CONSTANT!!!
parameter['Q_pot'] = 500e3 # potential heat per weight of binder in J/kg

# additional function values
time = 60*60*24*28
dt = 60*30
T = 25

# ============================================================================ #
#                              Set numeric values                              #
# ============================================================================ #

# 'true' value of a, and its normal prior parameters
a_true = 2.5
loc_a = 2.0
scale_a = 1.0

# 'true' value of b, and its normal prior parameters
b_true = 1.7
loc_b = 1.0
scale_b = 1.0

# 'true' value of additive error sd, and its uniform prior parameters
sigma = 0.5
low_sigma = 0.1
high_sigma = 0.8

# the number of generated experiment_names and seed for random numbers
n_tests = 50
seed = 1



problem = InferenceProblem("Linear regression with normal additive error")

problem.add_parameter(
    "a",
    "model",
    tex="$a$",
    info="Slope of the graph",
    prior=("normal", {"loc": loc_a, "scale": scale_a}),
)
problem.add_parameter(
    "b",
    "model",
    info="Intersection of graph with y-axis",
    tex="$b$",
    prior=("normal", {"loc": loc_b, "scale": scale_b}),
)
problem.add_parameter(
    "sigma",
    "likelihood",
    tex=r"$\sigma$",
    info="Standard deviation, of zero-mean additive model error",
    prior=("uniform", {"low": low_sigma, "high": high_sigma}),
)

linear_model = LinearModel()
problem.add_forward_model("LinearModel", linear_model)

# data-generation; normal likelihood with constant variance around each point
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = linear_model.response(
    {linear_model.input_sensor.name: x_test, "m": a_true, "b": b_true}
)[linear_model.output_sensor.name]
y_test = np.random.normal(loc=y_true, scale=sigma)

# add the experimental data
problem.add_experiment(
    f"TestSeries_1",
    fwd_model_name="LinearModel",
    sensor_values={
        linear_model.input_sensor.name: x_test,
        linear_model.output_sensor.name: y_test,
    },
)
plot = True
# plot the true and noisy data
if plot:
    plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
    plt.plot(x_test, y_true, label="true", c="black")
    plt.xlabel(linear_model.input_sensor.name)
    plt.ylabel(linear_model.output_sensor.name)
    plt.legend()
    plt.tight_layout()
    plt.draw()  # does not stop execution
plt.show()

# add the noise model to the problem
problem.add_likelihood_model(
    GaussianLikelihoodModel(
        prms_def={"sigma": "std_model"}, sensors=linear_model.output_sensors[0]
    )
)

# give problem overview
problem.info()

# ============================================================================ #
#                    Solve problem with inference engine(s)                    #
# ============================================================================ #

# this routine is imported from another script because it it used by all
# integration tests in the same way; ref_values are used for plotting
true_values = {"a": a_true, "b": b_true, "sigma": sigma}

n_steps = 200
n_initial_steps = 100
n_walkers = 20
plot = False
show_progress = False
run_scipy= True
run_emcee = True
run_torch = True
run_dynesty = True


run_inference_engines(
    problem,
    true_values=true_values,
    n_steps=n_steps,
    n_initial_steps=n_initial_steps,
    n_walkers=n_walkers,
    plot=plot,
    show_progress=show_progress,
    run_scipy=run_scipy,
    run_emcee=run_emcee,
    run_torch=run_torch,
    run_dynesty=run_dynesty,
)
