import numpy as np



parameter_prior = 0.5
hyperparameter_z = np.random.binomial(1,parameter_prior,size=1) 

if hyperparameter_z == 1:
    parameter = np.random.lognormal(np.log(210), np.sqrt(np.log(1))) #np.random.normal(0,1)
else:
    parameter = 0

print(parameter)

