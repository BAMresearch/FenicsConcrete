import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import matplotlib.pyplot as matplot
import math

#########################################################################
#########################################################################
#########################################################################
#1st Step - Data Generation
#########################################################################
#########################################################################
#########################################################################

p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['problem'] =  'tensile_test'    #'tensile_test' #'bending_test' 
p['degree'] = 1
p['num_elements_length'] = 50
p['num_elements_breadth'] = 10
p['dim'] = 2
# Uncertainty type:
# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs.
# 3: Torsion Springs

p['nu'] = 0.28 
p['uncertainties'] = [0]
p['k_x'] = 1e12
p['k_y'] = 1e12
#p['K_torsion'] = 1e11

p['constitutive'] = 'isotropic'
p['E_m'] = 1.72228206e+07 
p['E_d'] = 1.96009718e+08 
p['nu_12'] = 0.28
p['G_12'] = 210e6/(2*(1+0.28))    #p['E_m']/(2*(1+p['nu_12']))


# N/m², m, kg, sec, N
#p['length'] = 5
#p['breadth'] = 1
#p['load'] = [1e6,0] 
#p['rho'] = 7750
#p['g'] = 9.81
#p['E'] = 210e9 


# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa

p['dirichlet_bdy'] = 'left'

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
#problem.add_sensor(fenicsX_concrete.sensors.ReactionForceSensor())
problem.solve() 
#reaction_force_data = problem.sensors['ReactionForceSensor'].data[-1]
#displacement_data = problem.displacement.x.array
problem.pv_plot("Displacement_cantilever_x.xdmf")

problem.p.load = [0, 1e3]
problem.p.dirichlet_bdy = 'bottom'
experiment.p.dirichlet_bdy = 'bottom'
problem.experiment.bcs = problem.experiment.create_displ_bcs(problem.experiment.V)
problem.apply_neumann_bc()
problem.solve() 
problem.pv_plot("Displacement_cantilever_y.xdmf")