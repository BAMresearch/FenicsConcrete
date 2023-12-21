
import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
import json 

#with open('test_config.json', 'r') as f: 
#    json_object = json.loads(f.read()) 

# Adding sensors to the problem definition.
def add_sensor(length, breadth, _problem, _dirichlet_bdy, _sensors_num_edge_hor, _sensors_num_edge_ver): 
    sensor = []
    if _dirichlet_bdy == 0: #'left'
        for i in range(_sensors_num_edge_hor): 
            #print((p['length']*(i+1))/_sensors_num_edge_hor) #p['length']
            x_coord = (length*(i+1))/_sensors_num_edge_hor
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[x_coord, 0, 0]]), 'top')) #1/20
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[x_coord, breadth, 0]]), 'bottom'))
        
        for i in range(_sensors_num_edge_ver):
            #print((p['breadth']*(i+1))/(_sensors_num_edge_ver+1))
            y_coord = (breadth*(i+1))/(_sensors_num_edge_ver+1)
            sensor.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[length, y_coord, 0]]), 'right'))

        for i in range(len(sensor)):
            _problem.add_sensor(sensor[i])
        return len(sensor)
    
    
def run_test(exp, prob, dirichlet_bdy, load, sensor_flag = 0):
    prob.solve()
    prob.pv_plot("Displacement.xdmf")
    #store_sensor_data(prob)
    if sensor_flag == 1:
        counter=0
        displacement_at_sensors = np.zeros((len(prob.sensors),2))
        for i in prob.sensors:
            displacement_at_sensors[counter] = prob.sensors[i].data[-1]
            counter += 1
        #prob.sensors = fenicsX_concrete.sensors.Sensors()
        return displacement_at_sensors#.flatten()
    elif sensor_flag == 0:
        return prob.displacement.x.array


p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['degree'] = 1
p['num_elements_z'] = 150
p['num_elements_x'] = int(0.5/20*p['num_elements_z'])+1
p['num_elements_y'] = int(1/20*p['num_elements_z'])+1#4
#p['num_elements_z'] = 30#20
p['dim'] = 3
# Uncertainty type:
# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs. 
# 3: Torsion Springs
p['uncertainties'] = [0]
#p['k_x'] = 0.5e7
#p['k_y'] = 0.5e7

p['constitutive'] = 'isotropic' #'orthotropic' 
p['nu'] = 0.#0.28


# Kgmms⁻2/mm², mm, kg, sec, N
p['dim_x'] = 0.5#0.5
p['dim_y'] = 1#0.05
p['dim_z'] = 20.#1.
p['load'] = [0, 2e7, 0] #[1e3, 0] 
p['lower_limit_x'] = 0.5*(p['dim_x'] - 0.1)  #0*p['dim_x']
p['upper_limit_x'] = 0.5*(p['dim_x'] + 0.1) + 1e-5  #p['dim_x']
p['lower_limit_z'] = 0.5*(p['dim_z'] - 0.1)  #0.8*p['dim_z']
p['upper_limit_z'] = 0.5*(p['dim_z'] + 0.1)  #p['dim_z']
p['E'] = 1e5#210e9 #200e6 #Kgmms⁻2/mm² 


p['body_force'] = False
p['rho'] = 7750 #7750e-9 #kg/mm³
p['g'] = 9.81 #9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['weight'] = [0, -p['rho']*p['g'], 0] #Kgmms⁻2/mm²
#sensors_num_edge_hor = 5
#sensors_num_edge_ver = 4

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

#Adding sensors to the problem definition.
#test1_sensors_total_num = add_sensor(p['length'], p['breadth'], problem, 0, sensors_num_edge_hor, sensors_num_edge_ver)
#sensor_positions = np.zeros((test1_sensors_total_num, 3))
#counter = 0
#for i in problem.sensors:
#    sensor_positions[counter] = problem.sensors[i].where[0]
#    counter += 1

problem.solve()
problem.solve_eigenvalue_problem()
#problem.pv_plot("Displacement.xdmf")
problem.pv_eigenvalue_plot()
#print("works till here")
#test1_data = run_test(experiment, problem, 0, p['load'] , 1)

#import dolfinx as df
##from dolfinx.fem.petsc import assemble_matrix
#help(df.fem.petsc)