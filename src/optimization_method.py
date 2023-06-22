from abc import ABC

import numpy as np
from numpy import ndarray


class OptimizationMethod(ABC):
    def __init__(self, learning_rate=0.1):
        self._learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    def adjust(self, delta: ndarray[float], data: ndarray[float], index: int, epoch: int) -> ndarray[float]:
        raise NotImplementedError()


class GradientDescentOptimization(OptimizationMethod):
    def adjust(self, delta: ndarray[float], data: ndarray[float], _, __) -> ndarray[float]:
        return self._learning_rate * np.dot(data.T, delta)


class MomentumOptimization(OptimizationMethod):
    def __init__(self, alpha=0.3, learning_rate=0.1):
        super().__init__(learning_rate)
        self._alpha = alpha
        self._prev = []

    def adjust(self, delta: ndarray[float], data: ndarray[float], index: int, _) -> ndarray[float]:
        while index >= len(self._prev):
            self._prev.append(0)

        self._prev[index] = self._learning_rate * np.dot(data.T, delta) + self._alpha * self._prev[index]

        return self._prev[index]


class AdamOptimization(OptimizationMethod):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__()
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._momentum = []
        self._rmsProp = []

    def adjust(self, delta: ndarray[float], data: ndarray[float], index: int, epoch: int) -> ndarray[float]:
        assert len(self._momentum) == len(self._rmsProp)

        while index >= len(self._momentum):
            self._momentum.append(0)
            self._rmsProp.append(0)

        gradient = - np.dot(data.T, delta)  # TODO: chequear si va este menos
        self._momentum[index] = self._beta_1 * self._momentum[index] + (1 - self._beta_1) * gradient
        self._rmsProp[index] = self._beta_2 * self._rmsProp[index] + (1 - self._beta_2) * np.power(gradient, 2)

        m = np.divide(self._momentum[index], 1 - self._beta_1 ** (epoch+1))
        v = np.divide(self._rmsProp[index], 1 - self._beta_2 ** (epoch+1))

        return -self._alpha * np.divide(m, (np.sqrt(v) + self._epsilon))
