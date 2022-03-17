import dolfin as df
import numpy as np

# sensor template
class Sensor:
    def measure(self, u):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class DisplacementSensor(Sensor):
    def __init__(self, where):
        self.where = where
        self.data = []

    def measure(self, problem, t=1.0):
        # get displacements
        self.data.append(np.concatenate(([t],problem.displacement(self.where))))


class TemperatureSensor(Sensor):
    # temperature sensor in celsius
    def __init__(self, where):
        self.where = where
        self.data = []

    def measure(self, problem, t=1.0):
        T = problem.temperature(self.where) - problem.p.zero_C
        self.data.append([t,T])


class MaxTemperatureSensor(Sensor):
    def __init__(self):
        self.data = [[0,0]]

    def measure(self, problem, t=1.0):
        max_T = np.amax(problem.temperature.vector().get_local()) - problem.p.zero_C
        if max_T > self.data[0][1]:
    	    self.data[0] = [t,max_T]


class DOHSensor(Sensor):
    def __init__(self, where):
        self.where = where
        self.data = []

    def measure(self, problem, t=1.0):
        # get DOH
        # TODO: problem with projected field onto linear mesh!?!
        alpha = problem.degree_of_hydration(self.where)
        self.data.append([t,alpha])


class MinDOHSensor(Sensor):
    def __init__(self):
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get min DOH
        min_DOH = np.amin(problem.q_degree_of_hydration.vector().get_local())
        self.data.append([t,min_DOH])
        
        
# TODO: add more sensor for other fields

