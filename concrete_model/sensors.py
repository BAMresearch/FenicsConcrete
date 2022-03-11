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
        self.data.append(np.concatenate(([t],problem.mechanics_problem.u(self.where))))


class TemperatureSensor(Sensor):
    # temperature sensor in celsius
    def __init__(self, where):
        self.where = where
        self.data = []

    def measure(self, problem, t=1.0):
        T = problem.temperature_problem.T(self.where) - problem.temperature_problem.zero_C
        self.data.append([t,T])


class MaxTemperatureSensor(Sensor):
    def __init__(self):
        self.data = []

    def measure(self, problem, t=1.0):
        max_T = np.amax(problem.temperature_problem.T.vector().get_local()) - problem.temperature_problem.zero_C
        self.data.append([t,max_T])


class DOHSensor(Sensor):
    def __init__(self, where):
        self.where = where
        self.data = []

    def measure(self, problem, t=1.0):
        # get DOH
        alpha_projected = df.project(problem.temperature_problem.q_alpha, problem.temperature_problem.visu_space)
        alpha = alpha_projected(self.where)
        self.data.append([t,alpha])


class MinDOHSensor(Sensor):
    def __init__(self):
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        # get min DOH
        min_DOH = np.amin(problem.temperature_problem.q_alpha.vector().get_local())
        self.data.append([t,min_DOH])

