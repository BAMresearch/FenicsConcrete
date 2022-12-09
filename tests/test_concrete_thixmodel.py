import os

import dolfin as df
import numpy as np
import pytest

import fenics_concrete


def setup_test(parameters, sensor):
    experiment = fenics_concrete.ConcreteCubeUniaxialExperiment(parameters)

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"

    problem = fenics_concrete.ConcreteAMMechanical(
        experiment,
        parameters,
        mech_prob_string="ConcreteThixElasticModel",
        pv_name=file_path + "test_thix",
    )
    if parameters["bc_setting"] == "disp":
        problem.experiment.apply_displ_load(parameters["u_bc"])  # full
    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])
    # problem.add_sensor(sensor)

    # set time step
    problem.set_timestep(problem.p.dt)  # for time integration scheme

    # all elements same layer
    path_exp = df.Expression("tt", degree=1, tt=0)  # default!
    problem.set_initial_path(path_exp)

    return problem


def test_displ_thix_3D():
    """
    uniaxial tension test displacement control to check thixotropy material class
    sample is pulled twice first at time step 0 and second in the middle of the time; each step by u_bc
    """
    parameters = fenics_concrete.Parameters()  # using the current default values

    parameters["dim"] = 3  # 2
    parameters["mesh_density"] = 2
    parameters["log_level"] = "INFO"
    parameters["density"] = 0.0
    parameters["u_bc"] = 0.1
    parameters["bc_setting"] = "disp"
    parameters["age_0"] = 10  # s # age of concrete at expriment start time
    parameters["nu"] = 0.2

    parameters["time"] = 30 * 60  # total simulation time in s
    parameters["dt"] = 0.5 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.5, 1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.5, 1))

    prob3D = setup_test(parameters, [sensor01, sensor02])

    # solve
    E_o_time = []
    # define load increments
    dubcs = np.zeros(int(parameters["time"] / parameters["dt"]) + 1)
    dubcs[0] = 1
    dubcs[int(len(dubcs) / 2)] = 1  # second loading
    i = 0
    t = 0  # initialize time
    # solve
    while t <= prob3D.p.time:  # time
        # set load increment
        prob3D.experiment.apply_displ_load(dubcs[i] * parameters["u_bc"])
        i += 1

        # solve
        prob3D.solve(t=t)  # solving this
        prob3D.pv_plot(t=t)

        # store Young's modulus
        if t + parameters["age_0"] <= prob3D.p.t_f:
            E_o_time.append(prob3D.p.E_0 + prob3D.p.R_E * (t + parameters["age_0"]))
        else:
            E_o_time.append(
                prob3D.p.E_0
                + prob3D.p.R_E * prob3D.p.t_f
                + prob3D.p.A_E * (t + parameters["age_0"] - prob3D.p.t_f)
            )

        # prepare next timestep
        t += prob3D.p.dt

    # tests
    # get stresses and strains over time in zz
    sig_o_time = np.array(prob3D.sensors[sensor01.name].data)[:, -1]
    eps_o_time = np.array(prob3D.sensors[sensor02.name].data)[:, -1]
    # strain perpendicular to loading direction [xx and yy]
    epsxx_o_time = np.array(prob3D.sensors[sensor02.name].data)[:, 0]
    epsyy_o_time = np.array(prob3D.sensors[sensor02.name].data)[:, 4]

    # check strain at first and second loading
    assert eps_o_time[0] == pytest.approx(prob3D.p.u_bc)  # L==1!
    assert eps_o_time[-1] == pytest.approx(2 * prob3D.p.u_bc)  # L==1!
    assert epsxx_o_time[0] == pytest.approx(-prob3D.p.nu * prob3D.p.u_bc)
    assert epsxx_o_time[-1] == pytest.approx(-prob3D.p.nu * 2 * prob3D.p.u_bc)
    assert epsyy_o_time[0] == pytest.approx(-prob3D.p.nu * prob3D.p.u_bc)
    assert epsyy_o_time[-1] == pytest.approx(-prob3D.p.nu * 2 * prob3D.p.u_bc)

    # expected stress value compare to computed stress [check E modul evaluation]
    assert sig_o_time[0] == pytest.approx(parameters["u_bc"] / 1 * E_o_time[0])
    assert sig_o_time[-1] == pytest.approx(
        sum(E_o_time * dubcs * parameters["u_bc"] / 1)
    )  # E(0)*u_bc + E(2.loading)*u_bc

    # check if between loading step nothing changes
    assert len(np.where(np.diff(sig_o_time) > 1e-6)[0]) == pytest.approx(1)
    assert len(np.where(np.diff(eps_o_time) > 1e-6)[0]) == pytest.approx(1)


def test_displ_thix_2D():
    """
    uniaxial tension test displacement control to check thixotropy material class
    sample is pulled twice first at time step 0 and second in the middle of the time; each step by u_bc
    """
    parameters = fenics_concrete.Parameters()  # using the current default values

    parameters["dim"] = 2
    parameters["mesh_density"] = 5
    parameters["log_level"] = "INFO"
    parameters["density"] = 0.0
    parameters["u_bc"] = 0.1
    parameters["bc_setting"] = "disp"
    parameters["age_0"] = 10  # s # age of concrete at expriment start time
    parameters["nu"] = 0.2
    parameters["stress_state"] = "plane_stress"

    parameters["time"] = 30 * 60  # total simulation time in s
    parameters["dt"] = 0.5 * 60  # 0.5 min step

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 1))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 1))

    prob2D = setup_test(parameters, [sensor01, sensor02])

    E_o_time = []
    # define load increments of bc
    dubcs = np.zeros(int(parameters["time"] / parameters["dt"]) + 1)
    dubcs[0] = 1  # first loading
    dubcs[int(len(dubcs) / 2)] = 1  # second loading
    i = 0
    t = 0  # initialize time
    # solve
    while t <= prob2D.p.time:  # time
        # set load increment u_bc (for density automatic!)
        prob2D.experiment.apply_displ_load(dubcs[i] * parameters["u_bc"])
        i += 1
        # solve
        prob2D.solve(t=t)
        prob2D.pv_plot(t=t)

        # store Young's modulus for checks
        if t + parameters["age_0"] <= prob2D.p.t_f:
            E_o_time.append(prob2D.p.E_0 + prob2D.p.R_E * (t + parameters["age_0"]))
        else:
            E_o_time.append(
                prob2D.p.E_0
                + prob2D.p.R_E * prob2D.p.t_f
                + prob2D.p.A_E * (t + parameters["age_0"] - prob2D.p.t_f)
            )

        # prepare next timestep
        t += prob2D.p.dt

    # tests
    # get stresses and strains over time in yy
    sig_o_time = np.array(prob2D.sensors[sensor01.name].data)[:, -1]
    eps_o_time = np.array(prob2D.sensors[sensor02.name].data)[:, -1]
    # strain perpendicular to loading direction [xx]
    epsxx_o_time = np.array(prob2D.sensors[sensor02.name].data)[:, 0]

    # print("stresses yy", sig_o_time)
    # print("strains yy", eps_o_time)
    # print("E_o_time", E_o_time)

    # check strain at first and second loading
    assert eps_o_time[0] == pytest.approx(prob2D.p.u_bc)  # L==1!
    assert eps_o_time[-1] == pytest.approx(2 * prob2D.p.u_bc)
    assert epsxx_o_time[0] == pytest.approx(-prob2D.p.nu * prob2D.p.u_bc)
    assert epsxx_o_time[-1] == pytest.approx(-prob2D.p.nu * 2 * prob2D.p.u_bc)

    # expected stress value compared to computed stress [check E modul evaluation]
    assert sig_o_time[0] == pytest.approx(parameters["u_bc"] / 1 * E_o_time[0])
    assert sig_o_time[-1] == pytest.approx(
        sum(E_o_time * dubcs * parameters["u_bc"] / 1)
    )  # E(0)*u_bc + E(2.loading)*u_bc

    # check if between loading step nothing changes
    assert len(np.where(np.diff(sig_o_time) > 1e-6)[0]) == pytest.approx(1)
    assert len(np.where(np.diff(eps_o_time) > 1e-6)[0]) == pytest.approx(1)


@pytest.mark.parametrize("R_E", [0, 10e4])
@pytest.mark.parametrize("factor", [1, 3])
def test_density_thix_2D(R_E, factor):
    """
    uniaxial tension test with density without change in Young's modulus over time
    checking general implementation
    """
    parameters = fenics_concrete.Parameters()  # using the current default values

    parameters["dim"] = 2
    parameters["mesh_density"] = 5
    parameters["degree"] = 2
    parameters["log_level"] = "INFO"
    parameters["density"] = 2070.0
    parameters["bc_setting"] = "density"
    parameters["age_0"] = 0  # s # age of concrete at expriment start time
    parameters["nu"] = 0.2
    parameters["stress_state"] = "plane_stress"

    parameters["E_0"] = 2070000
    parameters["R_E"] = R_E  # if 0 no change in time!
    parameters["A_E"] = 0
    parameters["t_f"] = 5 * 60  # > time -> will not reached!

    parameters["time"] = 4 * 60  # total simulation time in s
    parameters["dt"] = 1 * 60  # 0.5 min step
    parameters["load_time"] = (
        factor * parameters["dt"]
    )  # load applied in factor x time step

    # sensor
    # 1.strainsensor middle bottom
    sensor01 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.0))
    # 2.strainsensor middle middle
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.5))
    # force sensor
    sensor03 = fenics_concrete.sensors.ReactionForceSensorBottom()
    # stress sensor middle bottom
    sensor04 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.0))
    # displacmenet sensor middle top
    sensor05 = fenics_concrete.sensors.DisplacementSensor(df.Point(0.5, 1.0))

    prob2D = setup_test(parameters, [sensor01, sensor02, sensor03, sensor04, sensor05])

    # solve
    E_o_time = []
    # initialize time
    t = 0
    while t <= prob2D.p.time:  # time
        # solve
        prob2D.solve(t=t)
        prob2D.pv_plot(t=t)

        # store Young's modulus for checks
        if t + parameters["age_0"] <= prob2D.p.t_f:
            E_o_time.append(prob2D.p.E_0 + prob2D.p.R_E * (t + parameters["age_0"]))
        else:
            E_o_time.append(
                prob2D.p.E_0
                + prob2D.p.R_E * prob2D.p.t_f
                + prob2D.p.A_E * (t + parameters["age_0"] - prob2D.p.t_f)
            )

        # prepare next timestep
        t += prob2D.p.dt

    # output over time steps in yy direction
    sig_o_time = np.array(prob2D.sensors[sensor04.name].data)[:, -1]
    eps_o_time = np.array(prob2D.sensors[sensor01.name].data)[:, -1]
    disp_o_time = np.array(prob2D.sensors[sensor05.name].data)[:, -1]
    force_o_time = prob2D.sensors[sensor03.name].data
    # print("E_o_time", E_o_time)
    # print("sig_o_time", sig_o_time)
    # print("eps_o_time", eps_o_time)
    # print("disp_o_time", disp_o_time)
    # print("force_o_time", force_o_time)

    # tests
    # print("force_bottom", np.sum(prob2D.sensors[sensor03.name].data))
    # print("dead load", -parameters["density"] * prob2D.p.g * 1 * 1)
    #
    # print(
    #     "strain analytic t=0",
    #     -1.0 / factor * parameters["density"] * prob2D.p.g / E_o_time[0],
    # )
    # print(
    #     "e ratio computed",
    #     sig_o_time[0] / eps_o_time[0],
    #     np.diff(sig_o_time) / np.diff(eps_o_time),
    # )
    # print("E ratio given", E_o_time[0] / E_o_time[1])
    # print("sig diff", np.diff(sig_o_time), sum(np.diff(sig_o_time)))

    # standard: dead load of full structure and strain
    force_bottom = np.sum(prob2D.sensors[sensor03.name].data)  # sum of all force values
    assert force_bottom == pytest.approx(-parameters["density"] * prob2D.p.g * 1 * 1)

    # strain at first time step
    assert eps_o_time[0] == pytest.approx(
        -1.0 / factor * parameters["density"] * prob2D.p.g / E_o_time[0], abs=1e-4
    )

    # check if stress changes accordingly to change in E_modul, if loading
    if factor > 1:  # if load in more steps applied
        # ratio sig/eps t=0 to sig/eps t=dt
        E_ratio_computed = (sig_o_time[0] / eps_o_time[0]) / (
            np.diff(sig_o_time)[0] / np.diff(eps_o_time)[0]
        )
        assert E_ratio_computed == pytest.approx(E_o_time[0] / E_o_time[1])
        # same delta sig in both steps
        if factor > 2:  # otherwise np.diff not long enough!
            assert np.diff(sig_o_time)[0] == pytest.approx(np.diff(sig_o_time)[1])
    else:
        # check that there are no changes in the stress
        assert sum(np.diff(sig_o_time)) == pytest.approx(0, abs=1e-8)
        assert sum(np.diff(eps_o_time)) == pytest.approx(0, abs=1e-8)


if __name__ == "__main__":

    # test_displ_thix_2D()
    #
    # test_displ_thix_3D()
    #
    # test_density_thix_2D(0)
    # test_density_thix_2D(10e4, 1)
    test_density_thix_2D(10e4, 3)
