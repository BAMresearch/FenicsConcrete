import os

import dolfin as df
import numpy as np
import pytest

import fenics_concrete


def setup_test_2D(parameters, mech_prob_string, sensor):

    # general parameters
    parameters["mesh_density"] = 2
    parameters["log_level"] = "INFO"

    parameters["density"] = 0.0
    parameters["u_bc"] = 0.002  # == strain since dimensions 1!!
    parameters["bc_setting"] = "disp"

    parameters["E_0"] = 70e3
    parameters["E_1"] = 20e3
    parameters["eta"] = 2e3  # relaxation time: tau = eta/E_1
    parameters["nu"] = 0.3
    parameters["stress_state"] = "plane_strain"

    parameters["time"] = 1.5  # total simulation time in s
    parameters["dt"] = 0.01  # step (should be < tau=eta/E_1)

    # thixotropy parameter
    parameters["R_i"] = [80.0, 125.0, 3.0]
    parameters["A_i"] = [160.0, 255.0, 6.5]
    parameters["t_f"] = [0.5, 0.5, 0.5]
    parameters["age_0"] = 0.0

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
@pytest.mark.parametrize("mech_prob_string", ["ConcreteViscoDevElasticModel"])
@pytest.mark.parametrize("dim", [2, 3])
def test_relaxation(visco_case, mech_prob_string, dim):
    """
    uniaxial tension test displacement control to check relaxation of visco material class
    """
    parameters = fenics_concrete.Parameters()  # using the current default values

    # changing parameters:
    parameters["dim"] = dim
    parameters["visco_case"] = visco_case

    # sensor
    sensor01 = fenics_concrete.sensors.StressSensor(df.Point(1.0, 1.0))
    sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(1.0, 1.0))

    prop2D = setup_test_2D(parameters, mech_prob_string, [sensor01, sensor02])

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
    if prop2D.p.visco_case.lower() == "cmaxwell":
        sig0 = prop2D.p.E_0 * eps_r + prop2D.p.E_1 * eps_r
        sigend = prop2D.p.E_0 * eps_r
    elif prop2D.p.visco_case.lower() == "ckelvin":
        sig0 = prop2D.p.E_0 * eps_r
        sigend = (prop2D.p.E_0 * prop2D.p.E_1) / (prop2D.p.E_0 + prop2D.p.E_1) * eps_r
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

    ##### plotting #######
    # plot analytic 1D solution against computed (for relaxation test -> fits if nu=0 and small enough time steps)
    sig_yy = []
    tau = prop2D.p.eta / prop2D.p.E_1
    if prop2D.p.visco_case.lower() == "cmaxwell":
        for i in time:
            sig_yy.append(
                prop2D.p.E_0 * eps_r + prop2D.p.E_1 * eps_r * np.exp(-i / tau)
            )
    elif prop2D.p.visco_case.lower() == "ckelvin":
        for i in time:
            sig_yy.append(
                prop2D.p.E_0
                * eps_r
                / (prop2D.p.E_1 + prop2D.p.E_0)
                * (
                    prop2D.p.E_1
                    + prop2D.p.E_0
                    * np.exp(-i / tau * (prop2D.p.E_0 + prop2D.p.E_1) / prop2D.p.E_1)
                )
            )

    print("analytic 1D == 2D with nu=0", sig_yy)
    print("stress over time", sig_o_time)

    import matplotlib.pyplot as plt

    plt.plot(time, sig_yy, "*r", label="analytic")
    plt.plot(time, sig_o_time, "og", label="FEM")
    plt.legend()
    plt.show()


# if __name__ == "__main__":

#
#     # test_relaxation("cmaxwell", "ConcreteViscoDevElasticModel", 2)
#
#     test_relaxation("ckelvin", "ConcreteViscoDevElasticModel", 2)
#
#     # test_relaxation('ckelvin', 'ConcreteViscoDevThixElasticModel', 2)
