import fenics_concrete
import pytest

def test_thermal_homogenization():
    #input values
    # matrix
    matrix_E = 10 # MPa
    matrix_poissions_ratio = 0.2
    matrix_compressive_strength = 10
    matrix_thermal_conductivity = 15

    # aggregates
    aggregate_E = 30
    aggregate_poissions_ratio = 0.25
    aggregate_radius = 2 # mm
    aggregate_vol_frac = 0.3 # volume fraction
    aggregate_thermal_conductivity = 100

    # itz assumptions
    itz_thickness = 0.2 #mm
    itz_factor = 0.8 # percentage of stiffness of matrix

    homgenized_concrete_1 = fenics_concrete.ConcreteHomogenization(E_matrix=matrix_E,
                                                                   nu_matrix=matrix_poissions_ratio,
                                                                   fc_matrix=matrix_compressive_strength,
                                                                   kappa_matrix=matrix_thermal_conductivity)

    homgenized_concrete_2 = fenics_concrete.ConcreteHomogenization(E_matrix=matrix_E,
                                                                   nu_matrix=matrix_poissions_ratio,
                                                                   fc_matrix=matrix_compressive_strength,
                                                                   kappa_matrix=matrix_thermal_conductivity)
    # adding uncoated
    homgenized_concrete_1.add_uncoated_particle(E=aggregate_E,
                                                nu=aggregate_poissions_ratio,
                                                volume_fraction=aggregate_vol_frac,
                                                kappa=aggregate_thermal_conductivity)
    # adding coated
    homgenized_concrete_2.add_coated_particle(E_inclusion=aggregate_E,
                                              nu_inclusion=aggregate_poissions_ratio,
                                              itz_ratio=itz_factor,
                                              radius=aggregate_radius,
                                              coat_thickness=itz_thickness,
                                              volume_fraction=aggregate_vol_frac,
                                              kappa=aggregate_thermal_conductivity)

    assert homgenized_concrete_1.kappa_eff == pytest.approx(homgenized_concrete_2.kappa_eff)
    assert homgenized_concrete_1.kappa_eff == pytest.approx(25.980861244019145)


def test_stiffness_homogenization():
        # input values
        # matrix
        matrix_E = 10  # MPa
        matrix_poissions_ratio = 0.2
        matrix_compressive_strength = 10

        # air
        air_E = 10  # MPa
        air_vol_frac = 0.2

        # aggregates
        aggregate_E = 30
        aggregate_poissions_ratio = 0.25
        aggregate_radius = 2  # mm
        aggregate_vol_frac = 0.3  # volume fraction

        # itz assumptions
        itz_thickness = 0.2  # mm
        itz_factor = 0.8  # percentage of stiffness of matrix

        # testing new code
        homgenized_concrete = fenics_concrete.ConcreteHomogenization(E_matrix=matrix_E,
                                                                     nu_matrix=matrix_poissions_ratio,
                                                                     fc_matrix=matrix_compressive_strength)
        # adding airpores
        homgenized_concrete.add_uncoated_particle(E=air_E, nu=matrix_poissions_ratio, volume_fraction=air_vol_frac)
        # adding agregates
        homgenized_concrete.add_coated_particle(E_inclusion=aggregate_E, nu_inclusion=aggregate_poissions_ratio,
                                                itz_ratio=itz_factor, radius=aggregate_radius,
                                                coat_thickness=itz_thickness, volume_fraction=aggregate_vol_frac)

        assert homgenized_concrete.E_eff == pytest.approx(13.156471830404511)
        assert homgenized_concrete.nu_eff == pytest.approx(0.21110139111362222)
        assert homgenized_concrete.fc_eff == pytest.approx(11.317889983420725)


def test_volume_averages():
    #input values
    # matrix
    E = 42 # MPa
    poissions_ratio = 0.3
    compressive_strength = 10
    matrix_C = 10
    matrix_rho = 30

    aggregate_vol_frac = 0.5 # volume fraction
    aggregate_radius = 10
    aggregate_C = 30
    aggregate_rho = 10

    # itz assumptions
    itz_thickness = 0.2 #mm
    itz_factor = 0.8 # percentage of stiffness of matrix

    homgenized_concrete_1 = fenics_concrete.ConcreteHomogenization(E_matrix=E,
                                                                   nu_matrix=poissions_ratio,
                                                                   fc_matrix=compressive_strength,
                                                                   rho_matrix = matrix_rho,
                                                                   C_matrix = matrix_C)

    homgenized_concrete_2 = fenics_concrete.ConcreteHomogenization(E_matrix=E,
                                                                   nu_matrix=poissions_ratio,
                                                                   fc_matrix=compressive_strength,
                                                                   rho_matrix = matrix_rho,
                                                                   C_matrix = matrix_C)
    # adding uncoated
    homgenized_concrete_1.add_uncoated_particle(E=E,
                                                nu=poissions_ratio,
                                                volume_fraction=aggregate_vol_frac,
                                                rho = aggregate_rho,
                                                C =aggregate_C)
    # adding coated
    homgenized_concrete_2.add_coated_particle(E_inclusion=E,
                                              nu_inclusion=poissions_ratio,
                                              itz_ratio=itz_factor,
                                              radius=aggregate_radius,
                                              coat_thickness=itz_thickness,
                                              volume_fraction=aggregate_vol_frac,
                                              rho = aggregate_rho,
                                              C =aggregate_C)

    assert homgenized_concrete_1.C_vol_eff == pytest.approx(homgenized_concrete_2.C_vol_eff)
    assert homgenized_concrete_1.rho_eff == pytest.approx(homgenized_concrete_2.rho_eff)
    assert homgenized_concrete_1.rho_eff == pytest.approx(20)
    assert homgenized_concrete_1.C_vol_eff == pytest.approx(300)


def test_heat_release():
        # input values
        E = 10  # MPa
        poissions_ratio = 0.2
        compressive_strength = 10
        aggregate_vol_frac = 0.5  # volume fraction
        rho = 42
        Q = 7

        homgenized_concrete = fenics_concrete.ConcreteHomogenization(E_matrix=E,
                                                                     nu_matrix=poissions_ratio,
                                                                     fc_matrix=compressive_strength,
                                                                     rho_matrix = rho,
                                                                     Q_matrix=Q)
        # testing computation of heat release wrt volume
        assert homgenized_concrete.Q_vol_eff ==  pytest.approx(Q*rho)

        # adding aggregates
        homgenized_concrete.add_uncoated_particle(E=E, nu=poissions_ratio, volume_fraction=aggregate_vol_frac, rho=rho)

        assert homgenized_concrete.Q_vol_eff ==  pytest.approx(Q*rho*aggregate_vol_frac)