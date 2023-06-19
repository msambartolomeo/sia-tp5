from abc import ABC

import numpy as np
from numpy import ndarray


class CutCondition(ABC):
    def is_finished(self, errors: ndarray[float]) -> bool:
        raise NotImplementedError()


class FalseCutCondition(CutCondition):
    def is_finished(self, errors: ndarray[float]) -> bool:
        return False


class AccuracyCutCondition(CutCondition):
    def is_finished(self, errors: ndarray[float]) -> bool:
        result = np.count_nonzero(np.logical_not(errors)) == len(errors)

        return result


class AbsoluteValueCutCondition(CutCondition):
    def is_finished(self, errors) -> bool:
        return np.sum(np.abs(errors)) == 0


class MSECutCondition(CutCondition):
    def __init__(self, eps: float = 0.01):
        self._eps = eps

    def is_finished(self, errors) -> bool:
        return np.average(errors ** 2) < self._eps


class OneWrongPixelCutCondition(CutCondition):
    def is_finished(self, errors: ndarray[float]) -> bool:
        for row in errors:
            if len(row) - np.count_nonzero(np.isclose(row, 0, atol=0.01)) > 1:
                return False

        return True
