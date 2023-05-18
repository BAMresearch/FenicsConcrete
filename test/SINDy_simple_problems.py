
import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
#print(parentdir)
import numpy as np
import fenicsX_concrete
from scipy import optimize
import matplotlib.pyplot as matplot
import pysindy as ps
import math

# Problem 1 
# Equation to recognise: f' = 2 + x - f
# f = 1 + x + 10*np.exp(-x)

""" start=0
end=3
number=300
x = np.linspace(start, end, number)
f = 1 + x + 10*np.exp(-x)
h=(end-start)/number

#Including boundaries
#f_bar = np.zeros(number)
#f_bar[0] = (f[1]-f[0])/h
#f_bar[number-1] = (f[number-1]-f[number-2])/h  
#for i in range(1,number-1):
#    f_bar[i] = (f[i+1]-f[i-1])/(2*h)
#
#F_matrix = np.zeros((number, 4))
#F_matrix[:,0] = 1
#F_matrix[:,1] = x
##F_matrix[:,2] = x**2
#F_matrix[:,3] = f
##F_matrix[:,4] = f**2
#LHS_vector = np.copy(f_bar) 


#Not including boundaries
f_bar = np.zeros(number-2) 
counter =0
for i in range(1,number-1):
    f_bar[counter] = (f[i+1]-f[i-1])/(2*h)
    counter += 1

number_of_functions = 3
F_matrix = np.zeros((number-2, number_of_functions))
F_matrix[:,0] = 2*np.ones(number-2)
F_matrix[:,1] = x[1:number-1]
F_matrix[:,2] = f[1:number-1]
#F_matrix[:,1] = np.cos(x[1:number-1])
#F_matrix[:,3] = x[1:number-1]**2
#F_matrix[:,4] = f[1:number-1]**3
#F_matrix[:,5] = x[1:number-1]**5
LHS_vector = np.copy(f_bar) - 2

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(F_matrix, LHS_vector)
print(clf.coef_)

#clf2 = linear_model.Ridge(alpha=0.0)
#clf2.fit(F_matrix, LHS_vector)
#print(clf2.coef_) 

data = np.stack((x, f), axis=-1)
optimiser = ps.STLSQ(threshold=0.07, fit_intercept=False)
feature_lib=feature_library=ps.PolynomialLibrary(degree=2) #ps.FourierLibrary(n_frequencies=3)
model = ps.SINDy(feature_names=['f', 'x']) #feature_library=feature_lib

print(model)
model.fit(f, u=x,t=x)
#print(model.get_feature_names())
model.print()
 """
############################################################################################################################################
# Problem 2
# Equation to recognise: f' = 2 + cos(x)
# f = 1 + x + 2*sin(x)

start=0
end=3
number=300
x = np.linspace(start, end, number)
f = 1 + x + 2*np.sin(x)
h=(end-start)/number


#Including boundaries
f_bar = np.zeros(number)
f_bar[0] = (f[1]-f[0])/h
f_bar[number-1] = (f[number-1]-f[number-2])/h  
for i in range(1,number-1):
    f_bar[i] = (f[i+1]-f[i-1])/(2*h)

#Not including boundaries
#f_bar = np.zeros(number-2) 
#counter =0
#for i in range(1,number-1):
#    f_bar[counter] = (f[i+1]-f[i-1])/(2*h)
#    counter += 1

number_of_functions = 3
F_matrix = np.zeros((number-2, number_of_functions))
F_matrix[:,0] = 2*np.ones(number-2)
F_matrix[:,1] = x[1:number-1]
F_matrix[:,2] = f[1:number-1]
LHS_vector = np.copy(f_bar) - 2

#from sklearn import linear_model
#clf = linear_model.Lasso(alpha=0.1)
#clf.fit(F_matrix, LHS_vector)
#print(clf.coef_)

#clf2 = linear_model.Ridge(alpha=0.0)
#clf2.fit(F_matrix, LHS_vector)
#print(clf2.coef_) 


optimiser = ps.STLSQ(threshold=0.07, fit_intercept=False)
feature_library_1=feature_library=ps.PolynomialLibrary(degree=2) #ps.FourierLibrary(n_frequencies=3)
feature_library_2=feature_library=ps.FourierLibrary(n_frequencies=1) 
model = ps.SINDy(feature_names=['f', 'x'], feature_library = feature_library_1 + feature_library_2) #feature_library=feature_lib

print(model)
model.fit(f, u=x,t=x)
print(model.get_feature_names())
model.print() 


#########################################################################################################################################
#reg = 1 / np.linalg.norm(x, 2, axis=0)
#            x_normed = x * reg

# Define PDE library that is quadratic in u, and 
# second-order in spatial derivatives of u.
#model.fit(f, u=x,t=x)
#x = np.linspace(0, 10)
#library_functions = [lambda x: x]
#library_function_names = [lambda x: x]
#pde_lib = ps.PDELibrary(
#    library_functions=library_functions,
#    function_names=library_function_names,
#    derivative_order=2,
#    spatial_grid=x,
#).fit([u])
#print("2nd order derivative library with function names: ")
#print(pde_lib.get_feature_names(), "\n")
