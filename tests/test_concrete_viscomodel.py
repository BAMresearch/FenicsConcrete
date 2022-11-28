import os

import dolfin as df
import numpy as np
import pytest

import fenics_concrete


def time_parameters(cur_t, parameters):
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
        parameters["nu"] = 0.3
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
        parameters["nu"] = 0.3
        parameters["stress_state"] = "plane_strain"

        # thixotropy parameter for (E_0, E_1, eta)
        parameters["R_i"] = {"E_0": 70.0e1, "E_1": 20.0e1, "eta": 2.0e1}
        parameters["A_i"] = {"E_0": 30.0e1, "E_1": 10.0e1, "eta": 1.0e1}
        parameters["t_f"] = {"E_0": 0.5, "E_1": 0.5, "eta": 0.5}
        parameters["age_0"] = 0.0

    else:
        raise ValueError("material type not implemented")

    return parameters


def setup_test_2D(parameters, mech_prob_string, sensor, mtype):

    # general parameters
    parameters["mesh_density"] = 2
    parameters["log_level"] = "INFO"

    parameters["density"] = 0.0
    parameters["u_bc"] = 0.002  # == strain because unit-square/cube (H=1)!!
    parameters["bc_setting"] = "disp"

    parameters["time"] = 1.5  # total simulation time in s
    parameters["dt"] = 0.01  # step (should be < tau=eta/E_1)

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
@pytest.mark.parametrize("mech_prob_string", ["ConcreteViscoDevThixElasticModel"])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
def test_relaxation(visco_case, mech_prob_string, dim, mtype):
    """
    uniaxial tension test displacement control to check relaxation of visco-thix material class
    """
    parameters = fenics_concrete.Parameters()  # using the current default values

    # changing parameters:
    parameters["dim"] = dim
    parameters["visco_case"] = visco_case

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(1.0, 1.0))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(1.0, 1.0))

    prop2D = setup_test_2D(parameters, mech_prob_string, [sensor01, sensor02], mtype)

    time = []
    # define load increments of bc fully applied in one step (alternative as time dependent dolfin Expression)
    dubcs = np.zeros(int(parameters["time"] / parameters["dt"]) + 1)
    dubcs[0] = 1
    i = 0
    # initialize time and solve!
    t = 0
    while t <= prop2D.p.time:  # time
        time.append(t)
        # set load increment u_bc (for density automatic!)
        prop2D.experiment.apply_displ_load(dubcs[i] * parameters["u_bc"])
        i += 1
        # solve
        prop2D.solve(t=t)  # solving this
        prop2D.pv_plot(t=t)
        # prepare next timestep
        t += prop2D.p.dt

    # get stress over time
    if prop2D.p.dim == 2:
        # sig_yy and eps_yy in case dim=2
        sig_o_time = np.array(prop2D.sensors[sensor01.name].data)[:, 1]
        # eps_o_time = np.array(prop2D.sensors[sensor02.name].data)[:,1]
    elif prop2D.p.dim == 3:
        # sig_zz and eps_zz in case dim=3
        sig_o_time = np.array(prop2D.sensors[sensor01.name].data)[:, 2]
        # eps_o_time = np.array(prop2D.sensors[sensor02.name].data)[:,2]

    # relaxation check - first and last value
    eps_r = prop2D.p.u_bc  # L==1 -> u_bc = eps_r (prescriped strain)
    #
    # print(prop2D.p.visco_case)
    # in case of thix model material parameters are changing over time!
    E_0_end, E_1_end, tau_end = time_parameters(time[-1], parameters)
    if prop2D.p.visco_case.lower() == "cmaxwell":
        sig0 = prop2D.p.E_0 * eps_r + prop2D.p.E_1 * eps_r
        sigend = E_0_end * eps_r
    elif prop2D.p.visco_case.lower() == "ckelvin":
        sig0 = prop2D.p.E_0 * eps_r
        sigend = (E_0_end * E_1_end) / (E_0_end + E_1_end) * eps_r
    else:
        raise ValueError("visco case not defined")

    print("theory", sig0, sigend)
    print("computed", sig_o_time[0], sig_o_time[-1])
    assert (sig_o_time[0] - sig0) / sig0 < 1e-8
    assert (sig_o_time[-1] - sigend) / sigend < 1e-4

    # get stresses and strains at the end
    # print('stresses',prop2D.sensors[sensor01.name].data[-1])
    # print('strains',prop2D.sensors[sensor02.name].data[-1])
    if prop2D.p.dim == 2:
        strain_xx = prop2D.sensors[sensor02.name].data[-1][0]
        strain_yy = prop2D.sensors[sensor02.name].data[-1][1]
        assert strain_yy == pytest.approx(prop2D.p.u_bc)  # L==1!
        assert strain_xx == pytest.approx(-prop2D.p.nu * prop2D.p.u_bc)
    elif prop2D.p.dim == 3:
        strain_xx = prop2D.sensors[sensor02.name].data[-1][0]
        strain_yy = prop2D.sensors[sensor02.name].data[-1][1]
        strain_zz = prop2D.sensors[sensor02.name].data[-1][2]
        assert strain_zz == pytest.approx(prop2D.p.u_bc)  # L==1!
        assert strain_xx == pytest.approx(-prop2D.p.nu * prop2D.p.u_bc)
        assert strain_yy == pytest.approx(-prop2D.p.nu * prop2D.p.u_bc)

    # # analytic 1D solution (for relaxation test -> fits if nu=0 and small enough time steps)
    # sig_yy = []
    # if prop2D.p.visco_case.lower() == "cmaxwell":
    #     for i in time:
    #         # compute current parameters
    #         E_0, E_1, tau = time_parameters(i, parameters)
    #         sig_yy.append(E_0 * eps_r + E_1 * eps_r * np.exp(-i / tau))
    # elif prop2D.p.visco_case.lower() == "ckelvin":
    #     for i in time:
    #         # compute current parameters
    #         E_0, E_1, tau = time_parameters(i, parameters)
    #         sig_yy.append(
    #             E_0
    #             * eps_r
    #             / (E_1 + E_0)
    #             * (E_1 + E_0 * np.exp(-i / tau * (E_0 + E_1) / E_1))
    #         )
    #
    # print("analytic 1D == 2D with nu=0", sig_yy)
    # print("stress over time", sig_o_time)
    #
    # ##### plotting #######
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(time, sig_yy, "*r", label="analytic")
    # plt.plot(time, sig_o_time, "og", label="FEM")
    # plt.legend()
    # plt.show()


# if __name__ == "__main__":
#
#     # test_relaxation("cmaxwell", "ConcreteViscoDevElasticModel", 2, "pure_visco")
#
#     # test_relaxation("ckelvin", "ConcreteViscoDevElasticModel", 2, "pure_visco")
#     # # both equivalent
#     # test_relaxation("ckelvin", "ConcreteViscoDevThixElasticModel", 2, "pure_visco")
#
#     # test_relaxation("ckelvin", "ConcreteViscoDevThixElasticModel", 2, "visco_thixo")
#
#     test_relaxation("ckelvin", "ConcreteViscoDevThixElasticModel", 3, "pure_visco")
