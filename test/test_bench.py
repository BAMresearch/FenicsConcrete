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
#1st Step - Data Generation
#########################################################################
#########################################################################

p = fenicsX_concrete.Parameters()  # using the current default values
p['bc_setting'] = 'free'
p['problem'] =  'tensile_test'    #'tensile_test' #'bending_test' #bending+tensile_test
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


# Kgmms⁻2/mm², mm, kg, sec, N
p['length'] = 5000
p['breadth'] = 1000
p['load'] = [1e3, 0] 
p['rho'] = 7750e-9 #kg/mm³
p['g'] = 9.81e3 #mm/s² for units to be consistent g must be given in m/s².
p['E'] = 210e6 #Kgmms⁻2/mm² ---- N/mm² or MPa

p['dirichlet_bdy'] = 'left'
p['constitutive'] = 'isotropic'
p['uncertainties'] = [0]

def run_test(exp, prob, dirichlet_bdy, load):
    prob.p.dirichlet_bdy = dirichlet_bdy
    exp.p.dirichlet_bdy = dirichlet_bdy
    prob.p.load = load
    prob.experiment.bcs = prob.experiment.create_displ_bcs(prob.experiment.V)
    prob.apply_neumann_bc()
    prob.calculate_bilinear_form()
    prob.solve()
    prob.pv_plot("Displacement.xdmf")
    return prob.displacement.x.array

experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.
data = np.copy(run_test(experiment, problem, 'left', [1e3, 0]))

# Kgmms⁻2/mm², mm, kg, sec, N
p['constitutive'] = 'orthotropic'
p['uncertainties'] = [0,2]
p['E_m'] = 210e6
p['E_d'] = 0.
p['nu_12'] = 0.28 #0.3
p['G_12'] =  210e6/(2*(1+0.28)) #(0.5*1e5)/(1+0.3)
p['k_x'] = 3.015424e+09 
p['k_y'] = 9.999990e+11

scaler = 500e6
experiment = fenicsX_concrete.concreteSlabExperiment(p)         # Specifies the domain, discretises it and apply Dirichlet BCs
problem = fenicsX_concrete.LinearElasticity(experiment, p)      # Specifies the material law and weak forms.

trial = np.copy(run_test(experiment, problem, 'left', [1e3, 0]))

