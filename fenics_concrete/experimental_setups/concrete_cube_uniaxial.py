import dolfin as df

from fenics_concrete.experimental_setups.experiment import Experiment
from fenics_concrete.helpers import Parameters


class ConcreteCubeUniaxialExperiment(Experiment):
    def __init__(self, parameters=None):
        # initialize a set of "basic paramters" (for now...)
        p = Parameters()
        p["dim"] = 3  # default boundary setting
        p["mesh_density"] = 10  # default boundary setting
        p["mesh_setting"] = "left/right"  # default boundary setting
        p["bc_setting"] = "disp"  # two boundary cases:
        #'disp': allow transverse contraction/ displacement on top apply_disp_load(value)
        #'density': uniaxial with density load applied & allow transverse contraction
        #'force': force applied at top surface/ uniaxial bc in displacements
        p["stress_case"] = "plane_stress"
        p = p + parameters
        super().__init__(p)

        # initialize variable top_displacement
        self.top_displacement = df.Constant(0.0)

    def setup(self):
        # elements per spacial direction
        n = self.p.mesh_density
        if self.p.dim == 2:
            self.mesh = df.UnitSquareMesh(n, n, self.p.mesh_setting)
        elif self.p.dim == 3:
            self.mesh = df.UnitCubeMesh(n, n, n)
        else:
            print(f"wrong dimension {self.p.dim} for problem setup")
            exit()

    def create_temp_bcs(self, V):
        assert ValueError(
            "create_tmp_bc not yet implemented for concrete_cube_uniaxial test"
        )

    def create_displ_bcs(self, V):
        """define uniaxial displacement boundaries
        Parameters
        ----------
        V : function space

        Returns
        ----
        displ_bcs : list, A list of DirichletBC objects, defining the boundary conditions
        """
        displ_bcs = []

        # two cases: displacement controlled or density controlled

        if self.p.dim == 2:
            if self.p.bc_setting == "disp":
                displ_bcs.append(
                    df.DirichletBC(V.sub(1), self.top_displacement, self.boundary_top())
                )

            displ_bcs.append(
                df.DirichletBC(V.sub(1), df.Constant(0), self.boundary_bottom())
            )
            displ_bcs.append(
                df.DirichletBC(V.sub(0), df.Constant(0), self.boundary_left())
            )

        elif self.p.dim == 3:
            if self.p.bc_setting == "disp":
                displ_bcs.append(
                    df.DirichletBC(V.sub(2), self.top_displacement, self.boundary_top())
                )

            displ_bcs.append(
                df.DirichletBC(V.sub(2), df.Constant(0), self.boundary_bottom())
            )
            displ_bcs.append(
                df.DirichletBC(V.sub(0), df.Constant(0), self.boundary_left())
            )
            displ_bcs.append(
                df.DirichletBC(V.sub(1), df.Constant(0), self.boundary_front())
            )

        return displ_bcs

    def apply_displ_load(self, top_displacement):
        """Updates the applied displacement load
        Parameters
        ----------
        top_displacement : float
            Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression
        """
        if self.p.bc_setting == "disp":
            self.top_displacement.assign(df.Constant(top_displacement))
        else:
            assert ValueError("displacement load not set in parameters")
