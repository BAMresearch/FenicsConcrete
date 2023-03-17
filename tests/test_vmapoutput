from vmap4fenics import VMAP4Fenics
import fenics_concrete

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