import os

import dolfin as df
import numpy as np
import pytest

import fenics_concrete


def get_parameters(cur_t, parameters):
    # compute E_0, E_1, tau for current time

    age = parameters["age_0"] + cur_t
    E_0 = bilinear_thix(
        age,
        parameters["E_0"],
        parameters["R_i"]["E_0"],
        parameters["A_i"]["E_0"],
        parameters["t_f"]["E_0"],
    )
    E_1 = bilinear_thix(
        age,
        parameters["E_1"],
        parameters["R_i"]["E_1"],
        parameters["A_i"]["E_1"],
        parameters["t_f"]["E_1"],
    )
    eta = bilinear_thix(
        age,
        parameters["eta"],
        parameters["R_i"]["eta"],
        parameters["A_i"]["eta"],
        parameters["t_f"]["eta"],
    )

    return E_0, E_1, eta / E_1


def bilinear_thix(age, P0, R, A, tf):
    # evaluate bilinear function P(age) = P0 + R * age for age <= tf and P = P0 + R + tf + A * (age - tf) for age > tf

    if age <= tf:
        P = P0 + R * age
    else:
        P = P0 + R * tf + A * (age - tf)

    return P


def material_parameters(parameters, mtype=""):

    if mtype.lower() == "pure_visco":
        # viscoelastic parameters
        parameters["E_0"] = 70e3
        parameters["E_1"] = 20e3
        parameters["eta"] = 2e3  # relaxation time: tau = eta/E_1
        parameters["nu"] = 0.0  # 0.3
        parameters["stress_state"] = "plane_strain"

        # thixotropy parameter for (E_0, E_1, eta)
        parameters["R_i"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters["A_i"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters["t_f"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters["age_0"] = 0.0

    elif mtype.lower() == "visco_thixo":

        # viscoelastic parameters
        parameters["E_0"] = 70e3
        parameters["E_1"] = 20e3
        parameters["eta"] = 2e3  # relaxation time: tau = eta/E_1
        parameters["nu"] = 0.0
        parameters["stress_state"] = "plane_strain"

        # thixotropy parameter for (E_0, E_1, eta)
        # choose parameters so that changing in simulation time (creep, relaxtion) neglectable [very short!]
        parameters["R_i"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters["A_i"] = {"E_0": 70.0e1, "E_1": 20.0e1, "eta": 2.0e1}
        parameters["t_f"] = {"E_0": 0.0, "E_1": 0.0, "eta": 0.0}
        parameters[
            "age_0"
        ] = 200.0  # much bigger than simulation time to see thix effect!

    else:
        raise ValueError("material type not implemented")

    return parameters


def setup_test_2D(parameters, mech_prob_string, sensor, mtype):

    # general parameters
    parameters["mesh_density"] = 2
    parameters["log_level"] = "INFO"

    parameters["time"] = 1.5  # total simulation time in s
    parameters["dt"] = 0.01  # 0.005  # step (should be < tau=eta/E_1)
    parameters["load_time"] = parameters["dt"]

    # material
    parameters = material_parameters(parameters, mtype=mtype)

    # experiment
    experiment = fenics_concrete.ConcreteCubeUniaxialExperiment(parameters)
    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"

    problem = fenics_concrete.ConcreteAMMechanical(
        experiment,
        parameters,
        mech_prob_string=mech_prob_string,
        pv_name=file_path + f"test2D_visco_{mech_prob_string}",
    )

    if parameters["bc_setting"] == "disp":
        problem.experiment.apply_displ_load(parameters["u_bc"])
    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])
    # problem.add_sensor(sensor)

    # set time step
    problem.set_timestep(problem.p.dt)  # for time integration scheme

    return problem


@pytest.mark.parametrize("visco_case", ["Cmaxwell", "Ckelvin"])
@pytest.mark.parametrize(
    "mech_prob_string",
    ["ConcreteViscoDevThixElasticModel"],
)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
def test_relaxation(visco_case, mech_prob_string, dim, mtype, plot=False):
    """
    uniaxial tension test displacement control to check relaxation of visco-thix material class
    """
    parameters = fenics_concrete.Parameters()  # using the current default values

    # changing parameters:
    parameters["dim"] = dim
    parameters["visco_case"] = visco_case
    parameters["density"] = 0.0
    parameters["u_bc"] = 0.002  # == strain because unit-square/cube (H=1)!!
    parameters["bc_setting"] = "disp"

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(1.0, 1.0))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(1.0, 1.0))

    prob = setup_test_2D(parameters, mech_prob_string, [sensor01, sensor02], mtype)

    time = []
    # define load increments of bc fully applied in one step (alternative as time dependent dolfin Expression)
    dubcs = np.zeros(int(parameters["time"] / parameters["dt"]) + 1)
    dubcs[0] = 1
    i = 0
    # initialize time and solve!
    t = 0
    while t <= prob.p.time:  # time
        time.append(t)
        # set load increment u_bc (for density automatic!)
        prob.experiment.apply_displ_load(dubcs[i] * parameters["u_bc"])
        i += 1
        # solve
        prob.solve(t=t)  # solving this
        prob.pv_plot(t=t)
        # prepare next timestep
        t += prob.p.dt

    # get stress over time (Tensor format)
    if prob.p.dim == 2:
        # sig_yy and eps_yy in case dim=2
        sig_o_time = np.array(prob.sensors[sensor01.name].data)[:, -1]
        # eps_o_time = np.array(prob.sensors[sensor02.name].data)[:, -1]
    elif prob.p.dim == 3:
        # sig_zz and eps_zz in case dim=3
        sig_o_time = np.array(prob.sensors[sensor01.name].data)[:, -1]
        # eps_o_time = np.array(prob.sensors[sensor02.name].data)[:, -1]

    # relaxation check - first and last value
    eps_r = prob.p.u_bc  # L==1 -> u_bc = eps_r (prescriped strain)
    #
    # print(prob.p.visco_case)
    # analytic case just for CONSTANT parameters over time (otherwise analytic integration not valid)
    # for tests, it is assumed that the total test time is short (age effects neglectable!) but initial concrete age can change parameter!
    E_0, E_1, tau = get_parameters(0, parameters)
    if prob.p.visco_case.lower() == "cmaxwell":
        sig0 = E_0 * eps_r + E_1 * eps_r
        sigend = E_0 * eps_r
    elif prob.p.visco_case.lower() == "ckelvin":
        sig0 = E_0 * eps_r
        sigend = (E_0 * E_1) / (E_0 + E_1) * eps_r
    else:
        raise ValueError("visco case not defined")

    # print("theory", sig0, sigend)
    # print("computed", sig_o_time[0], sig_o_time[-1])
    assert (sig_o_time[0] - sig0) / sig0 < 1e-8
    assert (sig_o_time[-1] - sigend) / sigend < 1e-4

    # get stresses and strain tensors at the end
    # print("stresses", prob.sensors[sensor01.name].data[-1])
    # print("strains", prob.sensors[sensor02.name].data[-1])
    if prob.p.dim == 2:
        strain_xx = prob.sensors[sensor02.name].data[-1][0]
        strain_yy = prob.sensors[sensor02.name].data[-1][-1]
        assert strain_yy == pytest.approx(prob.p.u_bc)  # L==1!
        assert strain_xx == pytest.approx(-prob.p.nu * prob.p.u_bc)
    elif prob.p.dim == 3:
        strain_xx = prob.sensors[sensor02.name].data[-1][0]
        strain_yy = prob.sensors[sensor02.name].data[-1][4]
        strain_zz = prob.sensors[sensor02.name].data[-1][-1]
        assert strain_zz == pytest.approx(prob.p.u_bc)  # L==1!
        assert strain_xx == pytest.approx(-prob.p.nu * prob.p.u_bc)
        assert strain_yy == pytest.approx(-prob.p.nu * prob.p.u_bc)

    # full analytic 1D solution (for relaxation test -> fits if nu=0 and small enough time steps)
    sig_yy = []
    if prob.p.visco_case.lower() == "cmaxwell":
        for i in time:
            sig_yy.append(E_0 * eps_r + E_1 * eps_r * np.exp(-i / tau))
    elif prob.p.visco_case.lower() == "ckelvin":
        for i in time:
            sig_yy.append(
                E_0
                * eps_r
                / (E_1 + E_0)
                * (E_1 + E_0 * np.exp(-i / tau * (E_0 + E_1) / E_1))
            )

    # print("analytic 1D == 2D with nu=0", sig_yy)
    # print("stress over time", sig_o_time)

    ##### plotting #######
    if plot:

        import matplotlib.pyplot as plt

        plt.plot(time, sig_yy, "*r", label="analytic")
        plt.plot(time, sig_o_time, "og", label="FEM")
        plt.xlabel("time [s]")
        plt.ylabel("stress yy [Pa]")
        plt.legend()
        plt.show()


@pytest.mark.parametrize("visco_case", ["Cmaxwell", "Ckelvin"])
@pytest.mark.parametrize(
    "mech_prob_string",
    ["ConcreteViscoDevThixElasticModel"],
)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
def test_creep(visco_case, mech_prob_string, dim, mtype, plot=False):
    """
    uniaxial tension test with density load as stress control to check creep of visco(-thix) material class
    """

    parameters = fenics_concrete.Parameters()  # using the current default values

    # changing parameters:
    parameters["dim"] = dim
    parameters["visco_case"] = visco_case
    parameters["density"] = 207.0  # load controlled
    parameters["bc_setting"] = "density"

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.0))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.0))

    prob = setup_test_2D(parameters, mech_prob_string, [sensor01, sensor02], mtype)

    time = []
    # initialize time and solve!
    t = 0
    while t <= prob.p.time:  # time
        time.append(t)
        # solve
        prob.solve(t=t)  # solving this
        prob.pv_plot(t=t)
        # prepare next timestep
        t += prob.p.dt

    # get stress over time
    if prob.p.dim == 2:
        # sig_yy and eps_yy in case dim=2
        sig_o_time = np.array(prob.sensors[sensor01.name].data)[:, -1]
        eps_o_time = np.array(prob.sensors[sensor02.name].data)[:, -1]
    elif prob.p.dim == 3:
        # sig_zz and eps_zz in case dim=3
        sig_o_time = np.array(prob.sensors[sensor01.name].data)[:, -1]
        eps_o_time = np.array(prob.sensors[sensor02.name].data)[:, -1]

    # relaxation check - first and last value
    sig_c = sig_o_time[0]
    assert sig_c == pytest.approx(-prob.p.density * prob.p.g)
    #
    # print(prob.p.visco_case)
    # analytic case just for CONSTANT parameters over time (otherwise analytic integration not valid)
    # for tests, it is assumed that the total test time is short (age effects neglectable!) but initial concrete age can change parameter!
    E_0, E_1, tau = get_parameters(0, parameters)
    if prob.p.visco_case.lower() == "cmaxwell":
        eps0 = sig_c / E_0 * (1 - E_1 / (E_0 + E_1))
        epsend = sig_c / E_0
    elif prob.p.visco_case.lower() == "ckelvin":
        eps0 = sig_c / E_0
        epsend = sig_c / E_0 + sig_c / E_1
    else:
        raise ValueError("visco case not defined")

    print("theory", eps0, epsend)
    print("computed", eps_o_time[0], eps_o_time[-1])
    assert (eps_o_time[0] - eps0) / eps0 < 1e-8
    assert (eps_o_time[-1] - epsend) / epsend < 1e-4

    # analytic 1D solution (for creep test -> fits if nu=0 and small enough time steps)
    eps_yy = []
    if prob.p.visco_case.lower() == "cmaxwell":
        for i in time:
            eps_yy.append(
                sig_c
                / E_0
                * (1 - E_1 / (E_0 + E_1) * np.exp(-i / tau * E_0 / (E_0 + E_1)))
            )
    elif prob.p.visco_case.lower() == "ckelvin":
        for i in time:
            eps_yy.append(sig_c / E_0 + sig_c / E_1 * (1 - np.exp(-i / tau)))

    # print("analytic 1D == 2D with nu=0", eps_yy)
    # print("stress over time", eps_o_time)

    ##### plotting #######
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(time, eps_yy, "*r", label="analytic")
        plt.plot(time, eps_o_time, "og", label="FEM")
        plt.xlabel("time [s]")
        plt.ylabel("eps yy [-]")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    # extra test for standard visco elastic model (above all combinations for Thix-Visco Model)
    plot = False
    test_relaxation(
        "cmaxwell", "ConcreteViscoDevElasticModel", 2, "pure_visco", plot=plot
    )
    test_relaxation(
        "ckelvin", "ConcreteViscoDevElasticModel", 2, "pure_visco", plot=plot
    )

#     # test_creep("cmaxwell", "ConcreteViscoDevElasticModel", 2, "pure_visco", plot=True)
#     # test_creep("ckelvin", "ConcreteViscoDevElasticModel", 2, "pure_visco", plot=True)
#
#     # equivalent to "ConcreteDevElasticModel"
#     # test_relaxation(
#     #     "ckelvin", "ConcreteViscoDevThixElasticModel", 2, "pure_visco", plot=True
#     # )
#     # test_creep(
#     #     "ckelvin", "ConcreteViscoDevThixElasticModel", 2, "pure_visco", plot=True
#     # )
#
#     # with thixo effect
#     # test_relaxation(
#     #     "ckelvin", "ConcreteViscoDevThixElasticModel", 2, "visco_thixo", plot=True
#     # )
#     # test_creep(
#     #     "ckelvin", "ConcreteViscoDevThixElasticModel", 2, "visco_thixo", plot=True
#     # )
