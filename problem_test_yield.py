from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import concrete_experiment as concrete_experiment
import concrete_problem as concrete_problem

#import probeye
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.scipy_.solver import ScipySolver


import time

# initiate material problem
material_problem = concrete_problem.ConcreteThermoMechanical()

stress1 = [10,-30,20,10,0,0]
stress2 = [60,60,60,0,40,0]
stress3 = [50,0,50,0,-5,0]
stress4 = [-50,0,0,0,0,12]
stresses =  np.array([stress1,stress2,stress3,stress4])


for i in range(1):
     stresses = np.append(stresses,stresses, axis=0)


p_stress = material_problem.mechanics_problem.principal_stress(stresses)
print(p_stress)


# for i in range(1):
#     stresses = np.append(stresses,stresses, axis=0)
exit()


ft = np.empty(len(stresses))
fc = np.empty(len(stresses))
ft.fill(40)
fc.fill(60)



#
# #output = material_problem.mechanics_problem.rankine_yield(stress)
#
# print(output)


p_stress = material_problem.mechanics_problem.yield_surface_3(stresses, ft, fc)

print(p_stress)

exit()




max = 30

ft = 20
fc = 60
fc2 = 60


x_stress = np.arange(-80,70,1)
y_stress = np.arange(-80,70,1)
z = 0
x_len = len(x_stress)
y_len = len(y_stress)
data = np.zeros((x_len, y_len))
for i, x in enumerate(x_stress):
    for j,y in enumerate(y_stress):
        output = material_problem.mechanics_problem.yield_surface([x,y,z],ft,fc)
        if output > 0:
            data[i][j] = output+100
        print(x,y,output)

        #data.append([x,y,output])



#print(data)


plt.pcolor(x_stress,y_stress,data)

plt.tight_layout()
plt.show()



# get function over time!

