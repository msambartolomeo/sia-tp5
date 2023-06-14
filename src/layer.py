from numpy import ndarray


class Layer:
    def __init__(self, neurons: ndarray[float]):
        self._neurons = neurons

    @property
    def neurons(self) -> ndarray[float]:
        return self._neurons

    @neurons.setter
    def neurons(self, new_neurons: ndarray[float]):
        self._neurons = new_neurons
