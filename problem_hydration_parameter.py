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
from probeye.inference.scipy_.solver import ScipySolver
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
        self.parameters = ['eta']
        # irgendeine liste....
        self.input_sensors = [Sensor("T"),Sensor("dt"), Sensor("time_max"),  Sensor("E_act"), Sensor("Q_pot"), Sensor("T_ref"), Sensor("B1"), Sensor("B2"),Sensor("alpha_max"),Sensor("time")]
        self.output_sensors = [Sensor('time'),Sensor('heat')]

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

        # initiate material problem
        material_problem = concrete_problem.ConcreteThermoMechanical()
        # get the respective function
        hydration_fkt = material_problem.get_heat_of_hydration_ftk()

        time_list, heat_list, doh_list = hydration_fkt(T, time_max, dt, parameter)
        return {"time": time_list, 'heat': heat_list}



#------------------------------------------
# START PROBLEM DESCRIPTION!!!!!!!
#-------------------------------------------

# read data
T = 25
time_data = []
heat_data = []
with open(f'test_hydration_data_T{T}.dat') as f:
    for line in f:
        if line[0] != '#':
            vals = line.split()
            time_data.append(float(vals[0]))
            heat_data.append(float(vals[0]))

x_test = time_data
y_test = heat_data

#
# # initiate material problem
# material_problem = concrete_problem.ConcreteThermoMechanical()
# # get the respective function
# hydration_fkt = material_problem.get_heat_of_hydration_ftk()
#
# # set required parameter
# parameter = concrete_experiment.Parameters() # using the current default values
#
# parameter['B1'] = 2.916E-4  # in 1/s
# parameter['B2'] = 0.0024229  # -
# parameter['eta'] = 5.554  # something about diffusion
# parameter['alpha_max'] = 0.875  # also possible to approximate based on equation with w/c
# parameter['E_act'] = 47002   # activation energy in Jmol^-1
# parameter['T_ref'] = 25  # reference temperature in degree celsius
# parameter['igc'] = 8.3145  # ideal gas constant in [J/K/mol], CONSTANT!!!
# parameter['zero_C'] = 273.15  # in Kelvin, CONSTANT!!!
# parameter['Q_pot'] = 500e3 # potential heat per weight of binder in J/kg
#
# # additional function values
# time = 60*60*24*28
# dt = 60*30
# T = 25

# ============================================================================ #
#                              Set numeric values                              #
# ============================================================================ #

# 'true' value of a, and its normal prior parameters
eta_true = 5.554
loc_eta = 5.554 # starting guess
scale_eta = .01

# the number of generated experiment_names and seed for random numbers
n_tests = 10
seed = 1

problem = InferenceProblem("Linear regression with normal additive error")

problem.add_parameter(
    "eta",
    "model",
    tex=r"$\eta$",
    info="Some paramter, but important",
    prior=("normal", {"loc": loc_eta, "scale": scale_eta}),
)

problem.add_parameter(
    "sigma",
    "likelihood",
    tex=r"$\sigma",
    info="Some paramter, but important",
    const=0.01
)

hydration_heat_model = HydrationHeatModel()
problem.add_forward_model("HydrationHeatModel", hydration_heat_model)

# add the experimental data
problem.add_experiment(
    f"TestSeries_1",
    fwd_model_name="HydrationHeatModel",
    sensor_values={
        'time': x_test,
        'heat': y_test,
        'B1' : 2.916E-4, # in 1/s
    'B2' : 0.0024229,  # -
    'alpha_max': 0.875,  # also possible to approximate based on equation with w/c
    'E_act': 47002,   # activation energy in Jmol^-1
    'T_ref': 25,  # reference temperature in degree celsius
    'igc': 8.3145, # ideal gas constant in [J/K/mol], CONSTANT!!!
    'zero_C': 273.15,  # in Kelvin, CONSTANT!!!
    'Q_pot': 500e3, # potential heat per weight of binder in J/kg
    'T': 25,
    'dt': 60*30,
    'time_max': 60*60*24*28
    },

)
#
# plot = True
# # plot the true and noisy data
# if plot:
#     plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
#     plt.plot(x_test, y_true, label="true", c="black")
#     plt.xlabel(linear_model.input_sensor.name)
#     plt.ylabel(linear_model.output_sensor.name)
#     plt.legend()
#     plt.tight_layout()
#     plt.draw()  # does not stop execution
# plt.show()

# add the noise model to the problem
problem.add_likelihood_model(
    GaussianLikelihoodModel(
        prms_def={"sigma": "std_model"}, sensors=[hydration_heat_model.output_sensors[0], hydration_heat_model.output_sensors[1]]
    )
)

# give problem overview
problem.info()

scipy_solver = ScipySolver(problem)
inference_data = scipy_solver.run_max_likelihood()