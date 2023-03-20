from vmap4fenics import VMAP4Fenics
import PyVMAP as VMAP
import numpy as np
import fenics_concrete
from os.path import isfile
import pytest

def test_vmapoutput():
  # define problem
  parameters = fenics_concrete.Parameters()
  parameters['E'] = 30000
  parameters['nu'] = 0.2
  parameters['height'] = 300
  parameters['radius'] = 75
  parameters['mesh_density'] = 6
  parameters['log_level'] = 'WARNING'
  parameters['bc_setting'] = 'fixed'
  parameters['dim'] = 3
  experiment = fenics_concrete.ConcreteCylinderExperiment(parameters)
  problem = fenics_concrete.LinearElasticity(experiment = experiment, parameters = parameters, pv_name = 'test', vmapoutput = True)
  # define sensors
  problem.add_sensor(fenics_concrete.sensors.ReactionForceSensorBottom())
  problem.set_material(name = 'Linear_Concrete',
            state = 'solid',
            type = 'concrete',
            description= 'linear elastic model',
            material_id = 'linear_concrete_model',
            idealization = 'continuum',
            physics = 'solid mechanics'
            )

  # define displacement
  displacement_list = [1,5,10]
  # loop over all measured displacements
  for displacement in displacement_list:
    problem.experiment.apply_displ_load(displacement)
    problem.solve()  # solving this

  metaInfoRead = VMAP.sMetaInformation()
  unitSystemRead = VMAP.sUnitSystem()
  testVectorRead = VMAP.VectorTemplateCoordinateSystem()
  geomPointsRead = VMAP.sPointsBlock()
  geomElemsRead = VMAP.sElementBlock()
  elemTypesRead = VMAP.VectorTemplateElementType()
  intTypeRead = VMAP.VectorTemplateIntegrationType()
  variableRead = VMAP.VectorTemplateStateVariable()

  # test file exists
  assert isfile('test/test_VMAP.h5')

  # test metadata
  problem.wrapper.vmap_file.readMetaInformation(metaInfoRead)
  hasmeta = True
  if not metaInfoRead.getExporterName(): hasmeta = False
  if not metaInfoRead.getFileDate(): hasmeta = False
  if not metaInfoRead.getFileTime(): hasmeta = False
  if not metaInfoRead.getDescription(): hasmeta = False
  assert hasmeta

  # test unitsystem
  problem.wrapper.vmap_file.readUnitSystem(unitSystemRead)
  hasunits = True
  assert unitSystemRead.myLengthUnit.myUnitSymbol
  if not unitSystemRead.myMassUnit.myUnitSymbol: hasunits = False
  if not unitSystemRead.myTimeUnit.myUnitSymbol: hasunits = False
  if not unitSystemRead.myCurrentUnit.myUnitSymbol: hasunits = False
  if not unitSystemRead.myTemperatureUnit.myUnitSymbol: hasunits = False
  if not unitSystemRead.myAmountOfSubstanceUnit.myUnitSymbol: hasunits = False
  if not unitSystemRead.myLuminousIntensityUnit.myUnitSymbol: hasunits = False
  assert hasunits

  # test coorsystem
  problem.wrapper.vmap_file.readCoordinateSystems("/VMAP/SYSTEM",testVectorRead)
  coorsystem = np.array((testVectorRead[0].myIdentifier, testVectorRead[0].myType, testVectorRead[0].myReferencePoint, testVectorRead[0].myAxisVectors),dtype=VMAP.sCoordinateSystem)
  assert (coorsystem == np.array((1, VMAP.sCoordinateSystem.CARTESIAN_LEFT_HAND,  (0., 0., 0.), (1., 0., 0., 0., 1., 0., 0., 0., 1.)), dtype=VMAP.sCoordinateSystem)).all

  # test points
  problem.wrapper.vmap_file.readPointsBlock("/VMAP/GEOMETRY/1",geomPointsRead)
  points = problem.V.tabulate_dof_coordinates()[::problem.V.dofmap().block_size()].tolist()
  assert geomPointsRead.mySize == len(points)
  points_equal = True
  for i in range(geomPointsRead.mySize):
    geomCoords = [geomPointsRead.myCoordinates[3*i],geomPointsRead.myCoordinates[3*i+1],geomPointsRead.myCoordinates[3*i+2]]
    points_equal = points_equal and (geomCoords == points[i])
  assert(points_equal)

  # test integrationtype
  problem.wrapper.vmap_file.readIntegrationTypes(intTypeRead)
  assert intTypeRead[0].myIdentifier == VMAP.VMAPIntegrationTypeFactory.GAUSS_TETRAHEDRON_4

  # test elementtype
  problem.wrapper.vmap_file.readElementTypes(elemTypesRead)
  assert elemTypesRead[0].myShapeType == VMAP.sElementType.TETRAHEDRON_10
  integrationTypeExists = True
  for elemType in elemTypesRead:
    if not elemType.myIntegrationType in [intType.myIdentifier for intType in intTypeRead]: integrationTypeExists = False
  assert integrationTypeExists

  # test elements
  problem.wrapper.vmap_file.readElementsBlock("/VMAP/GEOMETRY/1",geomElemsRead)
  elemTypeExists = True

  for i in range(geomElemsRead.myElementsSize):
    if not geomElemsRead.getElement(i).myElementType in [elemType.myShapeType for elemType in elemTypesRead]: elemTypeExists = False
  assert elemTypeExists

  # dimension consistency check
  assert intTypeRead[0].myDimension == elemTypesRead[0].myDimension == len(geomPointsRead.myCoordinates)/geomPointsRead.mySize

  # test measurements
  allvaluescorrect = True
  problem.wrapper.vmap_file.readVariablesBlock("/VMAP/VARIABLES/STATE-1/1", variableRead)
  allvaluescorrect = allvaluescorrect and problem.sensors["ReactionForceSensorBottom"].data[0] == variableRead[0].myValues[0]
  problem.wrapper.vmap_file.readVariablesBlock("/VMAP/VARIABLES/STATE-2/1", variableRead)
  allvaluescorrect = allvaluescorrect and problem.sensors["ReactionForceSensorBottom"].data[1] == variableRead[0].myValues[0]
  problem.wrapper.vmap_file.readVariablesBlock("/VMAP/VARIABLES/STATE-3/1", variableRead)
  allvaluescorrect = allvaluescorrect and problem.sensors["ReactionForceSensorBottom"].data[2] == variableRead[0].myValues[0]
  assert allvaluescorrect