"""
    multiple layer geometry in 2D - plane strain
    - thix concrete model (FenicsConcrete)
    - pseudo density approach for activation
"""

import os

import dolfin as df
import numpy as np
import pytest

import fenics_concrete


def set_test_parameters(case=""):
    # define global parameters for layer tests
    #
    parameters = fenics_concrete.Parameters()  # using the current default values

    parameters["dim"] = 2
    parameters["stress_state"] = "plane_strain"
    parameters["degree"] = 1
    parameters["layer_number"] = 5  # changed in single layer test!!
    parameters["layer_height"] = 1 / 100  # m
    parameters["layer_width"] = 4 / 100  # m
    parameters["log_level"] = "INFO"
    parameters["density"] = 2070.0  # kg/m³
    # parameters['g'] = 9.81 # in material_problem.py default value

    parameters["t_layer"] = 20  # s time to build one layer # Wolf paper 0.31 min!
    parameters["dt"] = 5  # time step

    # 1. with linear elastic thixotropy model:
    if case.lower() == "thix":

        # material parameters similar as Wolfs et al 2018
        parameters["nu"] = 0.3  # Poissons Ratio
        parameters["E_0"] = 0.078e6  # Youngs Modulus Pa
        parameters["R_E"] = 0.0  # reflocculation rate of E modulus in Pa / s
        parameters["A_E"] = (
            100 * 0.0012e6 / 60
        )  # structuration rate of E modulus in Pa / s
        parameters["t_f"] = 0  # reflocculation time in s
        parameters["age_0"] = 0  # s concrete age at print head

        parameters["mech_prob_string"] = "ConcreteThixElasticModel"

    # 2. with linear elastic thixotropy model:
    elif case.lower() == "viscothix":

        # viscoelastic parameters (random visco parameters!!)
        parameters["E_0"] = 0.078e6  # Youngs Modulus Pa
        parameters["E_1"] = 0.1e6
        parameters["eta"] = 0.1e6  # relaxation time: tau = eta/E_1
        parameters["nu"] = 0.3

        # thixotropy parameter for (E_0, E_1, eta)
        parameters["R_i"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters["A_i"] = {"E_0": 100 * 0.0012e6 / 60, "E_1": 0.001e6, "eta": 0.0}
        parameters["t_f"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters["age_0"] = 0.0

        parameters["mech_prob_string"] = "ConcreteViscoDevThixElasticModel"
        parameters["visco_case"] = "CMaxwell"  # or "CKelvin"
        # for maxwell: nearly no effect by choosing tau big (E_1=0.1e6, eta=0.1e9)

    return parameters


def setup_problem(parameters, pv_name):
    # define problem
    experiment = fenics_concrete.ConcreteMultipleLayers2DExperiment(parameters)
    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    problem = fenics_concrete.ConcreteAMMechanical(
        experiment,
        parameters,
        mech_prob_string=parameters["mech_prob_string"],
        pv_name=file_path + pv_name,
    )

    # sensor
    problem.add_sensor(fenics_concrete.sensors.ReactionForceSensorBottom())
    problem.add_sensor(
        fenics_concrete.sensors.StrainSensor(
            df.Point(parameters["layer_width"] / 2, 0.0)
        )
    )
    problem.add_sensor(
        fenics_concrete.sensors.StressSensor(
            df.Point(parameters["layer_width"] / 2, 0.0)
        )
    )

    return problem


def define_path_time(prob, param, t_diff, t_0=0):
    # create path as layer wise at dof points and overwrite in prob
    # one layer by time
    """
    prob: problem
    param: parameter dictionary
    t_diff: time difference between each layer
    t_0: start time for all (0 if static computation)
                            (-end_time last layer if dynamic computation)
    """

    # extract default fct to be replaced
    tmp_path = prob.mechanics_problem.q_path
    # print(tmp_age.vector()[:])
    # dof map for coordinates
    dof_map = tmp_path.function_space().tabulate_dof_coordinates()[:]
    # print(tmp_age.function_space().dofmap().tabulate_all_coordinates())
    new_path = np.zeros(len(tmp_path.vector()[:]))
    y_CO = np.array(dof_map)[:, -1]
    h_min = np.arange(
        0, param["layer_number"] * param["layer_height"], param["layer_height"]
    )
    h_max = np.arange(
        param["layer_height"],
        (param["layer_number"] + 1) * param["layer_height"],
        param["layer_height"],
    )
    # print(y_CO)
    # print(h_min)
    # print(h_max)
    for i in range(0, len(h_min)):
        layer_index = np.where(
            (y_CO > h_min[i] - df.DOLFIN_EPS) & (y_CO <= h_max[i] + df.DOLFIN_EPS)
        )
        # print((parameters['layer_number']-i-1)*age_diff_layer)
        new_path[layer_index] = t_0 + (param["layer_number"] - 1 - i) * t_diff
    # print('new_path', new_path, new_path.min(), new_path.max())

    prob.mechanics_problem.q_path.vector()[:] = new_path[:]  # overwrite

    return prob


@pytest.mark.parametrize(
    "load_time_set",
    ["dt", "t_layer"],
)
def test_single_layer_2D(load_time_set):
    # One single layer build immediately and lying for a given time

    # set parameters
    parameters = set_test_parameters(
        case="thix"
    )  # just for thix implemented otherwise tests not working (analytic visco deformation missing)
    parameters["layer_number"] = 1
    parameters["age_0"] = 20
    parameters["degree"] = 2

    # incremental loading
    parameters["load_time"] = parameters[load_time_set]

    # set standard problem & sensor
    pv_name = "test_single_layer"
    problem = setup_problem(parameters, pv_name)

    # set time step
    dt = parameters["dt"]
    # total simulation time in s
    time = 2 * parameters["t_layer"]

    problem.set_timestep(dt)  # for time integration scheme
    # initialize time
    t = 0
    while t <= time:  # time
        # solve
        # print('solve for', t)
        problem.solve(t=t)
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

    # check results (multi-axial stress state not uniaxial no analytical stress solution)
    force_bottom = problem.sensors["ReactionForceSensorBottom"].data
    dead_load = (
        parameters["density"]
        * parameters["layer_width"]
        * parameters["layer_height"]
        * problem.p.g
    )
    # dead load of full structure
    assert sum(force_bottom) == pytest.approx(-dead_load)

    # check stress changes accordingly to chaneg in E_modul if loading applied
    # stress and strain in yy over time
    sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, -1]
    eps_o_time = np.array(problem.sensors["StrainSensor"].data)[:, -1]
    print("stress yy", sig_o_time)
    print("strain yy", eps_o_time)
    if load_time_set.lower() == "dt":
        # instance loading -> no further young's modulus changes
        assert sum(np.diff(sig_o_time)) == pytest.approx(0, abs=1e-8)
        assert sum(np.diff(eps_o_time)) == pytest.approx(0, abs=1e-8)
    elif load_time_set.lower() == "t_layer":
        print("check", np.diff(sig_o_time))
        # ratio sig/eps t=0 to sig/eps t=dt
        E_ratio_computed = (sig_o_time[0] / eps_o_time[0]) / (
            np.diff(sig_o_time)[0] / np.diff(eps_o_time)[0]
        )
        E_0 = problem.p.E_0 + problem.p.A_E * (0.0 + parameters["age_0"])
        if parameters["dt"] + parameters["age_0"] <= parameters["t_f"]:
            E_dt = problem.p.E_0 + problem.p.R_E * (
                parameters["dt"] + parameters["age_0"]
            )
        else:
            E_dt = (
                problem.p.E_0
                + problem.p.R_E * parameters["t_f"]
                + problem.p.A_E
                * (parameters["dt"] + parameters["age_0"] - parameters["t_f"])
            )

        assert E_ratio_computed == pytest.approx(E_0 / E_dt)
        # same delta stress in each load step
        assert np.diff(sig_o_time)[0] == pytest.approx(np.diff(sig_o_time)[1])

    else:
        print("load_time_set not known")


@pytest.mark.parametrize("mcase", ["thix", "viscothix"])
@pytest.mark.parametrize("factor", [1, 2])
def test_multiple_layers_2D(mcase, factor, plot=False):
    # several layers dynamically deposited with given path
    # whole layer activate at once after t_layer next layer...
    # incremental set up amount of density at once by factor!

    parameters = set_test_parameters(case=mcase)
    parameters["load_time"] = factor * parameters["dt"]

    # set standard problem & sensor
    pv_name = f"test_multilayer_{mcase}"
    problem = setup_problem(parameters, pv_name)

    # Layers given by path function
    path = df.Expression("0", degree=0)
    problem.set_initial_path(path)
    time_last_layer_set = (parameters["layer_number"] - 1) * parameters["t_layer"]
    problem = define_path_time(
        problem, parameters, parameters["t_layer"], t_0=-time_last_layer_set
    )

    # initialize time
    dt = parameters["dt"]
    time = (parameters["layer_number"]) * parameters[
        "t_layer"
    ]  # total simulation time in s
    problem.set_timestep(dt)
    t = 0
    while t <= time:  # time
        # solve temp-hydration-mechanics
        # print('solve for', t)
        problem.solve(t=t)  # solving this
        problem.pv_plot(t=t)
        # prepare next timestep
        t += dt

    # check residual force bottom
    force_bottom = problem.sensors["ReactionForceSensorBottom"].data[:]
    dead_load = (
        parameters["layer_number"]
        * parameters["density"]
        * parameters["layer_width"]
        * parameters["layer_height"]
        * problem.p.g
    )
    # print("force - weigth", force_bottom, sum(force_bottom), dead_load)
    assert sum(force_bottom) == pytest.approx(-dead_load)

    # # check E modulus evolution over structure (each layer different E)
    if mcase.lower() == "thix" and problem.p.R_E == 0:
        E_bottom_layer = problem.p.E_0 + problem.p.A_E * (time + parameters["age_0"])
        E_upper_layer = problem.p.E_0 + problem.p.A_E * (
            time
            - (parameters["layer_number"] - 1) * parameters["t_layer"]  # layers before
            + parameters["age_0"]
        )

        assert problem.mechanics_problem.q_E.vector()[:].min() == pytest.approx(
            E_upper_layer
        )
        assert problem.mechanics_problem.q_E.vector()[:].max() == pytest.approx(
            E_bottom_layer
        )
        # TODO: Emodulus sensor?

    if plot:
        # example plotting
        stress_yy = np.array(problem.sensors["StressSensor"].data)[:, -1]
        strain_yy = np.array(problem.sensors["StrainSensor"].data)[:, -1]
        # print("stress_yy", stress_yy)
        # print("strain_yy", strain_yy)
        # print("dstrain", np.diff(strain_yy))

        # # strain_yy over time
        import matplotlib.pylab as plt

        time_line = np.linspace(0, time, int(time / dt) + 1)
        plt.figure(1)
        plt.plot(time_line, strain_yy, "*-r")
        # plt.plot(time_line, stress_yy, "*-")
        plt.xlabel("process time")
        plt.ylabel("sensor bottom middle")
        plt.show()


if __name__ == "__main__":

    # # incremental loading:
    # # a) load applied immediately: parameters["load_time"] = parameters["dt"]
    # test_single_layer_2D("dt", "thix")
    # # b) load applied with in one layer time parameters["load_time"] = parameters["t_layer"]
    # test_single_layer_2D("t_layer", "thix")

    # test_multiple_layers_2D("thix", 1, plot=True)

    test_multiple_layers_2D("viscothix", 1, plot=True)
