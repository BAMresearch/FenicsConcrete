import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import matplotlib.pyplot as matplot
import math


p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['problem'] = 'tensile_test'      #'cantilever_beam' #
p['degree'] = 1
p['num_elements_length'] = 100
p['num_elements_breadth'] = 10
p['dim'] = 2
# Uncertainty type:
# 0: Constant E and nu fields.
# 1: Random E and nu fields.
# 2: Linear Springs.
# 3: Torsion Springs
p['uncertainties'] = [0]

p['constitutive'] = 'isotropic'
p['nu'] = 0.3 #0.28 #
p['E'] = 1e5  #210e9 #


p['length'] = 1
p['breadth'] = 0.1
p['load'] = [1000.,0]

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
problem.add_sensor(fenicsX_concrete.sensors.ReactionForceSensor())
problem.solve() 
problem.pv_plot("Displacement_iso.xdmf")