import dolfinx as df
#import sys
#print(sys.path)
from fenicsX_concrete.helpers import Parameters
from fenicsX_concrete.sensors import Sensors

from loguru import logger
import logging
import sys

import warnings


""" from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

df.parameters["form_compiler"]["representation"] = "quadrature"
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
 """

class MaterialProblem():
    def __init__(self, experiment, parameters=None, pv_name='pv_output_full'):
        self.experiment = experiment
        # setting up paramters
        self.p = Parameters()
        # constants
        # TODO: where to put these?, what about units???


        # other "globel" paramters...
        self.p['log_level'] = 'INFO'

        self.p = self.p + self.experiment.p + parameters

        # set log level...
        if self.p.log_level == 'NOTSET':
            df.log.LogLevel(0)
            logging.getLogger("FFC").setLevel(logging.NOTSET)
            logging.getLogger("UFL").setLevel(logging.NOTSET)
            logger.add(sys.stderr, level="NOTSET")
        elif self.p.log_level == 'DEBUG':
            df.log.LogLevel(10)
            logging.getLogger("FFC").setLevel(logging.DEBUG)
            logging.getLogger("UFL").setLevel(logging.DEBUG)
            logger.add(sys.stderr, level="DEBUG")
        elif self.p.log_level == 'INFO':
            df.log.LogLevel(20)
            logging.getLogger("FFC").setLevel(logging.INFO)
            logging.getLogger("UFL").setLevel(logging.INFO)
            logger.add(sys.stderr, level="INFO")
        elif self.p.log_level == 'WARNING':
            df.log.LogLevel(30)
            logging.getLogger("FFC").setLevel(logging.WARNING)
            logging.getLogger("UFL").setLevel(logging.WARNING)
            logger.add(sys.stderr, level="WARNING")
        elif self.p.log_level == 'ERROR':
            df.log.LogLevel(40)
            logging.getLogger("FFC").setLevel(logging.ERROR)
            logging.getLogger("UFL").setLevel(logging.ERROR)
            logger.add(sys.stderr, level="ERROR")
        elif self.p.log_level == 'CRITICAL':
            df.log.LogLevel(50)
            logging.getLogger("FFC").setLevel(logging.CRITICAL)
            logging.getLogger("UFL").setLevel(logging.CRITICAL)
            logger.add(sys.stderr, level="CRITICAL")
        else:
            level = self.p['log_level']
            raise Exception(f'unknown log level {level}')


        self.sensors =  Sensors()  # list to hold attached sensors


        self.pv_name = pv_name

        #setup fields for sensor output, can be defined in model
        #self.displacement = None
        #self.temperature = None
        #self.degree_of_hydration = None
        #self.q_degree_of_hydration = None

        # setup the material object to access the function
        self.setup()

    def setup(self):
        # initialization of this specific problem
        raise NotImplementedError()

    def solve(self):
        # define what to do, to solve this problem
        raise NotImplementedError()

    def add_sensor(self, sensor):
        self.sensors[sensor.name] = sensor

    def clean_sensor_data(self):
    #    #for sensor_object in self.sensors.values():
    #    #    sensor_object.data.clear()
        for i in self.sensors:
            self.sensors[i].data.clear()
            self.sensors[i].time.clear()


    def delete_sensor(self):
        del self.sensors
        self.sensors = Sensors()



        