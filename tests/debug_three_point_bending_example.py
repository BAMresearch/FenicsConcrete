import dolfin as df
import fenics_concrete
import matplotlib.pyplot as plt
import numpy as np
from paraview.simple import *

def three_point_bending_example(E, nu, pv_output=False):
    """Example of a linear elastic three point bending test

    Parameters
    ----------
        E : float
            Young's modulus in N/mm²
        nu : float
            Poisson's ratio

    Returns
    -------
        stress_in_x : float
            Stress in x direction in the center at the bottom, where the maximum stress is expected
    """
    # setting up the simulation parameters
    parameters = fenics_concrete.Parameters()  # using the current default values
    # input values for the material
    parameters['E'] = E
    parameters['nu'] = nu
    # definition of the beam and mesh
    parameters['dim'] = 3
    parameters['mesh_density'] = 4  # number of elements in vertical direction
    parameters['height'] = 300  # in mm
    parameters['length'] = 2000  # in mm
    parameters['width'] = 150  # in mm
    parameters['log_level'] = 'WARNING'
    # displacement load in the center of the beam
    displacement = -10  # displacement load in the center of the beam in mm

    # setting up the problem
    experiment = fenics_concrete.ConcreteBeamExperiment(parameters)
    problem = fenics_concrete.LinearElasticity(experiment, parameters, pv_name=f'linear_beam_E_{E}')

    # applying the load
    problem.experiment.apply_displ_load(displacement)

    # applying the stress sensor
    stress_sensor = fenics_concrete.sensors.StressSensor(df.Point(parameters.length/2, parameters.width/2, 0))
    #stress_sensor = fenics_concrete.sensors.StressSensor((parameters.length/2, parameters.width/2, 0))

    problem.add_sensor(stress_sensor)

    # solving the problem
    problem.solve(t=0)  # solving this

    if pv_output:
        problem.pv_plot(t=0)

    # results for last (only) load step
    stress_tensor = problem.sensors[stress_sensor.name].data[-1]
    stress_in_x = stress_tensor[0]


    return stress_in_x


def get_paraview_stress(filename, x, y, z):

    # create a new 'Xdmf3ReaderS'
    fem_model = Xdmf3ReaderS(FileName=[filename])
    fem_model.PointArrays = ['Stress']

    # create a new 'Probe Location'
    probeLocation = ProbeLocation(Input=fem_model, ProbeType='Fixed Radius Point Source')

    # init the 'Fixed Radius Point Source' selected for 'ProbeType'
    probeLocation.ProbeType.Center = [x, y, z]
    polyData = servermanager.Fetch(probeLocation)
    pointData = polyData.GetPointData()
    stressArray = pointData.GetArray('Stress')
    value = stressArray.GetTuple(0)
    return value



if __name__ == "__main__":
    # example of how to use this function
    # defining the material parameters

    #location
    x = 1000
    y = 75
    z = 0



    nu = 0.2
    stress = []
    pv_stress = []

    E_list = np.arange(10000, 15000, 1000).tolist()
    #E_list = [50000]
    for i, E in enumerate(E_list):
        print(f'{i}/{len(E_list)} Run E={E}')
        stress_x = three_point_bending_example(E, nu, pv_output=True)

        stress.append(stress_x)
        filename = f'linear_beam_E_{E}.xdmf'
        pv_stress.append(get_paraview_stress(filename, x, y, z)[0])

    # resulting stress in x direction in the bottom center of the beam

    plt.plot(E_list, stress, 'o')
    plt.plot(E_list, pv_stress, 'o')
    plt.show()
    #print(stress)
