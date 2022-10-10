import numpy as np
import pytest


def get_e_nu_from_k_g(K, G):
    E = 9*K*G/(3*K+G)
    nu = (3*K-2*G)/(2*(3*K+G))
    return E, nu


def get_k_g_from_e_nu(E, nu):
    K = E/(3*(1-2*nu))
    G = E/(2*(1+nu))
    return K, G


class ConcreteHomogenization():
    # object to compute homogenized parameters for cement matrix and aggregates
    def __init__(self, E_matrix, nu_matrix, fc_matrix, rho_matrix = 1, kappa_matrix = 1, C_matrix = 1, Q_matrix = 1):
        """ initializes the object

        matrix parameters are set

        Parameters
        ----------
        E_matrix : float
           Young's modulus of matrix material
        nu_matrix : float
            Poisson's Ratio of matrix material
        fc_matrix : float
            Conpressive strength of the matrix
        rho_matrix : float, optional
            Density of the matrix
        kappa_matrix : float, optional
            Thermal conductivity of the matrix
        C_matrix : float, optional
            Specific/volumetric heat capacity of the matrix
        Q_matrix : float, optional
            Heat release in Energy per Weight of binder
        """
        self.E_matrix = E_matrix
        self.nu_matrix = nu_matrix
        self.fc_matrix = fc_matrix
        self.kappa_matrix = kappa_matrix
        self.C_matrix = C_matrix
        self.rho_matrix = rho_matrix
        self.Q_matrix = Q_matrix
        self.vol_frac_matrix = 1
        self.vol_frac_binder = 1  # when coated inclusions are considered these still count as binder volume
        self.Q_eff = self.Q_matrix * self.rho_matrix * self.vol_frac_binder

        self.K_matrix, self.G_matrix = get_k_g_from_e_nu(E_matrix,nu_matrix)

        # initial values
        self.K_eff = self.K_matrix
        self.G_eff = self.G_matrix
        self.E_eff = E_matrix
        self.nu_eff = nu_matrix
        self.fc_eff = fc_matrix
        self.kappa_eff = kappa_matrix
        self.C_eff = C_matrix
        self.rho_eff = rho_matrix

        # list for inclusion values (all phases that are not matrix
        self.n_incl = 0
        self.vol_frac_incl = []

        self.A_dill_vol_incl = []
        self.A_dill_dev_incl = []
        self.G_incl = []
        self.K_incl = []
        self.A_therm_incl = []  # thermal conductivity
        self.kappa_incl = []
        self.C_incl = []
        self.rho_incl = []


        # auxiliary factors following Eshelby solution [Eshely, 1957]
        # required for uncoated computation
        self.alpha_0 = (1 + nu_matrix) / (3 * (1 + nu_matrix))
        self.beta_0 = 2 * (4 - 5 * nu_matrix) / (15 * (1 - nu_matrix))

    def add_uncoated_particle(self, E, nu, volume_fraction, rho = 1, kappa = 1, C = 1):
        """Adds a phase of uncoated material

        the particles are assumed to be homogeneous and spherical
        sets particle properties
        setup function called

        Parameters
        ----------
        E : float
            Young's modulus of particle material
        nu : float
            Poisson's Ratio of particle material
        volume_fraction : float
            Volume fraction of the particle within the composite
        rho : float, optional
            Density
        kappa : float, optional
            Thermal conductivity
        C : float, optional
            Specific/volumetric heat capacity

        """
        K, G = get_k_g_from_e_nu(E, nu)

        A_dil_vol = self.K_matrix / (self.K_matrix + self.alpha_0 * (K - self.K_matrix))
        A_dil_dev = self.G_matrix / (self.G_matrix + self.beta_0 * (G - self.G_matrix))

        # thermal concentration factor
        A_therm = 3 * self.kappa_matrix / (2 * self.kappa_matrix + kappa)

        # update global fields
        self.n_incl = self.n_incl + 1
        self.vol_frac_incl.append(volume_fraction)
        self.A_dill_vol_incl.append(A_dil_vol)
        self.A_dill_dev_incl.append(A_dil_dev)
        self.G_incl.append(G)
        self.K_incl.append(K)
        self.vol_frac_matrix = self.vol_frac_matrix - volume_fraction
        self.vol_frac_binder = self.vol_frac_binder - volume_fraction
        self.A_therm_incl.append(A_therm)
        self.kappa_incl.append(kappa)
        self.rho_incl.append(rho)
        self.C_incl.append(C)

        if self.vol_frac_matrix < 0:
            raise Exception('Volume fraction of matrix can not be smaller than zero!')

        self.update_effective_fields()


    def add_coated_particle(self, E_inclusion, nu_inclusion, itz_ratio, radius, coat_thickness,volume_fraction,
                            rho = 1, kappa = 1, C = 1):
        """Adds a phase of coated material

        the particles are assumed to be homogeneous and spherical, coated by degraded matrix material
        the computation is based on the formulation of Herve-Zaoui,???? and taken from the paper of
        .......,

        sets partilce and coating properties
        setup function called

        Parameters
        ----------
        E_inclusion : float
            Young's modulus of particle material
        nu_inclusion : float
            Poisson's Ratio of particle material
        itz_ratio :float
            value of the reduction of the stiffness of the material surrounding the particles
        radius : float
            Radius of particles
        coat_thickness : float
            Thickness of the coating
        volume_fraction : float
            Volume fraction of the particle within the composite
        rho : float, optional
            Density of the inclusion
        k : float, optional
            Thermal conductivity of the particle, the coat is ignored
        C : float, optional
            Specific/volumetric heat capacity of the inclusion
        """
        # set values - inclusion, coating, matrix
        E = np.array([E_inclusion, self.E_matrix*itz_ratio, self.E_matrix])
        nu = np.array([nu_inclusion, self.nu_matrix, self.nu_matrix])

        # list with radius for inclusion and coating
        R = np.array([radius, radius+coat_thickness])

        # compute volume fraction of itz     
        itz_vol_frac = (((radius + coat_thickness) / radius) ** 3 - 1) * volume_fraction
        
        


        # compute shear and bulk modulus
        # G = E / (2 * (1 + nu))  # G, shear modulus for the different phases
        # K = E / (3 * (1 - 2 * nu))  # K, bulk modulus for the different phases
        K, G = get_k_g_from_e_nu(E,nu)

        # for the influence on the overall stiffness three auxiliary factor are computed for each of the two phases
        # Q, A and B, to efficiently compute these many steps are computed first

        # initialize two axillary matrices N(2 x 2) and M(4 x 4) for coated particle, on for each of the two phases
        N = [np.empty((2, 2)), np.empty((2, 2))]
        M = [np.empty((4, 4)), np.empty((4, 4))]

        # loop over the two phases (inclusion, k=0 and coating, k=1)
        self.x_list = [0,0]
        for k in range(2):
            # more auxiliary variables a through f
            a = G[k] / G[k + 1] * (7 + 5 * nu[k]) * (7 - 10 * nu[k+1]) - (7 - 10 * nu[k]) * (7 + 5 * nu[k+1])
            b = G[k] / G[k+1] * (7 + 5 * nu[k]) + 4 * (7 - 10 * nu[k])
            c = (7 - 5 * nu[k+1]) + 2 * (4 - 5 * nu[k+1]) * G[k] / G[k+1]
            d = (7 + 5 * nu[k+1]) + 4 * (7 - 10 * nu[k+1]) * G[k] / G[k+1]
            e = 2 * (4 - 5 * nu[k]) + G[k] / G[k+1] * (7 - 5 * nu[k])
            f = (4 - 5 * nu[k]) * (7 - 5 * nu[k+1]) - G[k] / G[k+1] * (4 - 5 * nu[k+1]) * (7 - 5 * nu[k])
            alpha = G[k] / G[k+1] - 1

            self.x_list[k] = [G[k],G[k + 1],a, b, c, d, e, f, alpha]

            M[k][0][0] = c / 3
            M[k][0][1] = R[k] ** 2 * (3 * b - 7 * c) / (5 * (1 - 2 * nu[k]))
            M[k][0][2] = -12 * alpha / (R[k] ** 5)
            M[k][0][3] = 4 * (f - 27 * alpha) / (15 * R[k] ** 3 * (1 - 2 * nu[k]))
            M[k][1][0] = 0
            M[k][1][1] = b * (1 - 2 * nu[k+1]) / (7 * (1 - 2 * nu[k]))
            M[k][1][2] = -20 * alpha * (1 - 2 * nu[k+1]) / (7 * R[k] ** 7)
            M[k][1][3] = -12 * alpha * (1 - 2 * nu[k+1]) / (7 * R[k] ** 5 * (1 - 2 * nu[k]))
            M[k][2][0] = R[k] ** 5 * alpha / 2
            M[k][2][1] = -R[k] ** 7 * (2 * a + 147 * alpha) / (70 * (1 - 2 * nu[k]))
            M[k][2][2] = d / 7
            M[k][2][3] = R[k] ** 2 * (105 * (1 - nu[k+1]) + 12 * alpha * (7 - 10 * nu[k+1]) - 7 * e) / (35 * (1 - 2 * nu[k]))
            M[k][3][0] = -5 * alpha * R[k] ** 3 * (1 - 2 * nu[k+1]) / 6
            M[k][3][1] = 7 * alpha * R[k] ** 5 * (1 - 2 * nu[k+1]) / (2 * (1 - 2 * nu[k]))
            M[k][3][2] = 0
            M[k][3][3] = e * (1 - 2 * nu[k+1]) / (3 * (1 - 2 * nu[k]))
            # divide all by some factor
            M[k] = M[k] / (5 * (1 - nu[k+1]))

            N[k][0][0] = 3 * K[k] + 4 * G[k+1]
            N[k][0][1] = 4 / R[k] ** 3 * (G[k+1] - G[k])
            N[k][1][0] = 3 * R[k] ** 3 * (K[k+1] - K[k])
            N[k][1][1] = 3 * K[k+1] + 4 * G[k]

            N[k] = N[k] / (3 * K[k+1] + 4* G[k+1])

        # initialize more auxiliary fields Q, P, W, A, B
        Q = [[], []]
        P = [[], []]
        A = [[], []]
        B = [[], []]

        Q[0] = N[0]
        Q[1] = N[1].dot(N[0])

        P[0] = M[0]
        P[1] = M[1].dot(M[0])

        W = 1 / (P[1][1][1] * P[1][0][0] - P[1][0][1] * P[1][1][0]) * P[0].dot(
            np.array([[P[1][1][1]], [-P[1][1][0]], [0], [0]]))

        A[0] = P[1][1][1] / (P[1][1][1] * P[1][0][0] - P[1][0][1] * P[1][1][0])
        A[1] = W[0][0]

        B[0] = -P[1][1][0] / (P[1][1][1] * P[1][0][0] - P[1][0][1] * P[1][1][0])
        B[1] = W[1][0]

        # finally the required dillition factors, volumetric and deviatoric are computed!

        A_dil_vol_incl = 1 / Q[1][0][0]
        A_dil_vol_coat = Q[0][0][0] / Q[1][0][0]
        A_dil_dev_incl = A[0] - 21 / 5 * R[0] ** 2 / (1 - 2 * nu[0]) * B[0]
        A_dil_dev_coat = A[1] - 21 / 5 * (R[1] ** 5 - R[0] ** 5) / ((1 - 2 * nu[1]) * (R[1] ** 3 - R[0] ** 3)) * B[1]

        # thermal concentration factor
        # coating is set to matrix material
        A_therm = 3 * self.kappa_matrix / (2 * self.kappa_matrix + kappa)

        # update global fields
        # inclusion data
        self.vol_frac_incl.append(volume_fraction)
        self.A_dill_vol_incl.append(A_dil_vol_incl)
        self.A_dill_dev_incl.append(A_dil_dev_incl)
        self.G_incl.append(G[0])
        self.K_incl.append(K[0])
        self.A_therm_incl.append(A_therm)
        self.kappa_incl.append(kappa)
        self.rho_incl.append(rho)
        self.C_incl.append(C)
        # coating data
        self.vol_frac_incl.append(itz_vol_frac)
        self.A_dill_vol_incl.append(A_dil_vol_coat)
        self.A_dill_dev_incl.append(A_dil_dev_coat)
        self.G_incl.append(G[1])
        self.K_incl.append(K[1])
        self.A_therm_incl.append(1)  # coating is set to matrix material
        self.kappa_incl.append(self.kappa_matrix)
        self.rho_incl.append(self.rho_matrix)
        self.C_incl.append(self.C_matrix)
        # overall infos
        self.n_incl = self.n_incl + 2

        self.vol_frac_matrix = self.vol_frac_matrix - volume_fraction - itz_vol_frac
        self.vol_frac_binder = self.vol_frac_binder - volume_fraction

        if self.vol_frac_matrix < 0:
            raise Exception('Volume fraction of matrix can not be smaller than zero!')

        self.update_effective_fields()


    def update_effective_fields(self):
        K_eff_numerator = self.vol_frac_matrix * self.K_matrix
        K_eff_denominator = self.vol_frac_matrix
        G_eff_numerator = self.vol_frac_matrix * self.G_matrix
        G_eff_denominator = self.vol_frac_matrix
        kappa_eff_numerator = self.vol_frac_matrix * self.kappa_matrix
        kappa_eff_denominator = self.vol_frac_matrix
        self.rho_eff = self.vol_frac_matrix * self.rho_matrix
        self.C_eff = self.vol_frac_matrix * self.C_matrix
        vol_test = self.vol_frac_matrix

        for i in range(self.n_incl):

            K_eff_numerator += self.vol_frac_incl[i] * self.K_incl[i] * self.A_dill_vol_incl[i]
            K_eff_denominator += self.vol_frac_incl[i] * self.A_dill_vol_incl[i]
            G_eff_numerator += self.vol_frac_incl[i] * self.G_incl[i] * self.A_dill_dev_incl[i]
            G_eff_denominator += self.vol_frac_incl[i] * self.A_dill_dev_incl[i]
            kappa_eff_numerator += self.vol_frac_incl[i] * self.kappa_incl[i] * self.A_therm_incl[i]
            kappa_eff_denominator += self.vol_frac_incl[i] * self.A_therm_incl[i]
            self.rho_eff += self.vol_frac_incl[i] * self.rho_incl[i]
            self.C_eff += self.vol_frac_incl[i] * self.C_incl[i]
            vol_test += self.vol_frac_incl[i]

        assert vol_test == pytest.approx(1)  # sanity check that vol fraction have been corretly computed

            # compute effective properties
        self.K_eff = K_eff_numerator / K_eff_denominator
        self.G_eff = G_eff_numerator / G_eff_denominator
        self.E_eff = 9 * self.K_eff * self.G_eff / (3 * self.K_eff + self.G_eff)
        self.nu_eff = (3 * self.K_eff - 2 * self.G_eff) / (2 * (3 * self.K_eff + self.G_eff))

        self.kappa_eff = kappa_eff_numerator / kappa_eff_denominator

        # Mori-Tanaka factors for strength estimate
        A_MT_K = 1 / K_eff_denominator
        A_MT_G = 1 / G_eff_denominator

        ii = np.array([[1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])

        Iv = 1 / 3 * ii
        Id = np.eye(6) - Iv
        A_MT = A_MT_K * Iv + A_MT_G * Id # reversing volumetric and deviatoric split

        # compute material stiffness matrix L_eff (elastic) and matrix
        def L_from_k_and_g(k, g):
            ii = np.array([[1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

            Iv = 1 / 3 * ii
            Id = np.eye(6) - Iv
            L = 3 * k * Iv + 2 * g * Id

            return L

        L_matrix = L_from_k_and_g(self.K_matrix,self.G_matrix)
        L_eff = L_from_k_and_g(self.K_eff,self.G_eff)

        # todo continue with micro mech marble cement paper

        # calculation of compressive strength (J2 criterion for the matrix)
        sigma_0 = np.zeros((3, 3))
        sigma_0[0][0] = self.fc_matrix
        sigma_dev_0 = sigma_0-1/3*np.trace(sigma_0)*np.eye(3)
        j2_0 = np.sqrt(3/2*sum(sum(np.multiply(sigma_dev_0, sigma_dev_0))))
        strength_test = 1e-5  # "test" stress to be scaled
        sigma_test = np.array([[strength_test], [0], [0], [0], [0], [0]])

        B_MT = np.dot(np.dot(L_matrix, A_MT), np.linalg.inv(L_eff))
        s0 = np.dot(B_MT, sigma_test)
        stressInMatrix_tensor = np.array([[s0[0][0], s0[5][0], s0[4][0]],
                                          [s0[5][0], s0[1][0], s0[3][0]],
                                          [s0[4][0], s0[3][0], s0[2][0]]])
        sigma_dev_matrix = stressInMatrix_tensor - 1 / 3 * np.trace(stressInMatrix_tensor) * np.eye(3)
        j2_matrix = np.sqrt(3 / 2 * sum(sum(np.multiply(sigma_dev_matrix, sigma_dev_matrix))))
        fc = j2_0 / j2_matrix * strength_test
        self.fc_eff = fc

        # update heat release
        self.Q_eff = self.Q_matrix * self.rho_matrix * self.vol_frac_binder
