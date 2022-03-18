import dolfin as df
import numpy as np


from concrete_model.helpers import Parameters

class Experiment:
    def __init__(self, parameters = None):
        # setup of paramter field
        self.p = Parameters()
        # constants
        self.p['zero_C'] = 273.15  # to convert celcius to kelvin input to

        self.p = self.p + parameters

        self.setup()

    def setup(self):
        raise NotImplementedError()
        
  
