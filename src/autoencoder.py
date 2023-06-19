import numpy as np
from numpy import ndarray

from .multi_layer_perceptron import MultiLayerPerceptron


class Autoencoder(MultiLayerPerceptron):
    def get_latent(self, data: ndarray[float]) -> ndarray[float]:
        results = data
        for i in range(len(self._layers) // 2):
            results = np.insert(np.atleast_2d(results), 0, 1, axis=1)
            h = np.dot(results, self._layers[i].neurons)
            results = self._activation_function.evaluate(h)
        return results

    def create_from_latent(self, latent_data: ndarray[float]) -> ndarray[float]:
        results = latent_data
        for i in range(len(self._layers) // 2, len(self._layers)):
            results = np.insert(np.atleast_2d(results), 0, 1, axis=1)
            h = np.dot(results, self._layers[i].neurons)
            results = self._activation_function.evaluate(h)
        return results
