import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
import matplotlib.pyplot as matplot


p = fenicsX_concrete.Parameters()  # using the current default values
p['problem'] =  'tensile_test'   
# Choose bwtween 'tensile_test' and 'bending_test'
p['degree'] = 1
p['num_elements_length'] = 50
p['num_elements_breadth'] = 10
p['dim'] = 2

p['uncertainties'] = [0,2]
# Uncertainty type:
# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs. - Define the spring stiffnesses in k_x and k_y i.e. p['k_x'] and p['k_y'].
p['k_x'] = 1e12
p['k_y'] = 1e12

# 3: Torsion Springs - Define the spring stiffness in K_torsion i.e. p['K_torsion'].
#p['K_torsion'] = 1e11

p['constitutive'] = 'isotropic'
# Choose between:
# isotropic : Define E and nu.
p['E'] = 210e6
p['nu'] = 0.28 

# orthotropic: Define E_m, E_d, nu_12, G_12.
#p['E_m'] = 210e6 
#p['E_d'] = 0. 
#p['nu_12'] = 0.28
#p['G_12'] = p['E_m']/(2*(1+p['nu_12']))

# Stress in Kgmms⁻2/mm², Length in mm, Mass in kg, Time in sec
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa

p['dirichlet_bdy'] = 'left'

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
problem.solve() 
problem.pv_plot("Displacement_Results.xdmf")
