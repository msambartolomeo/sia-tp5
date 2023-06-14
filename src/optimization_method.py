from abc import ABC

import numpy as np
from numpy import ndarray


class OptimizationMethod(ABC):
    def __init__(self, learning_rate=0.1):
        self._learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    def adjust(self, delta: ndarray[float], data: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError()


class GradientDescentOptimization(OptimizationMethod):
    def adjust(self, delta: ndarray[float], data: ndarray[float]) -> ndarray[float]:
        return self._learning_rate * np.dot(data.T, delta)


class MomentumOptimization(OptimizationMethod):
    def __init__(self, alpha=0.3, learning_rate=0.1, architecture=None):
        super().__init__(learning_rate)
        self._alpha = alpha
        self._prev = [0]
        if architecture is not None:
            aux = []
            for i in range(len(architecture) - 1):
                aux.append(np.zeros((architecture[i] + 1, architecture[i + 1])))
            self._prev = []
            for i in reversed(range(len(aux))):
                self._prev.append(aux[i])

        self._index = 0

    def adjust(self, delta: ndarray[float], data: ndarray[float]) -> ndarray[float]:
        if self._index == len(self._prev):  # Aritmetica modular? No ubico esa calle pa
            self._index = 0

        self._prev[self._index] = self._learning_rate * np.dot(data.T, delta) + self._alpha * self._prev[self._index]
        self._index += 1
        return self._prev[self._index - 1]
