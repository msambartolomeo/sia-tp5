import numpy as np

from numpy import ndarray


def mse(errors: ndarray[float]) -> ndarray:
    return np.mean(errors ** 2)
