import pickle
from typing import List

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.error import mse
from src.layer import Layer
from src.optimization_method import OptimizationMethod


class MultiLayerPerceptron:
    def __init__(self, architecture: List[int], epochs: int, cut_condition: CutCondition,
                 activation_method: ActivationMethod, optimization_method: OptimizationMethod):

        self._epochs = epochs
        self._cut_condition = cut_condition
        self._activation_function = activation_method
        self._optimization_method = optimization_method

        # Initialize weights for the whole network with random [-1,1] values.
        self._layers = []
        for i in range(len(architecture) - 1):
            self._layers.append(Layer(np.random.uniform(-1, 1, (architecture[i] + 1, architecture[i + 1]))))

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        results = data
        for i in range(len(self._layers)):
            results = np.insert(np.atleast_2d(results), 0, 1, axis=1)
            # results = mu x hidden_size + 1, #layers[i] = (hidden_size + 1) x next_hidden_size
            h = np.dot(results, self._layers[i].neurons)
            # h = mu x next_hidden_size
            results = self._activation_function.evaluate(h)

        return results

    def feedforward(self, data):
        feedforward_data = [data]
        results = data
        feedforward_output = []
        for i in range(len(self._layers)):
            results = np.insert(results, 0, 1, axis=1)
            feedforward_output.append(results)
            # results = mu x hidden_size + 1, #layers[i] = (hidden_size + 1) x next_hidden_size
            h = np.dot(results, self._layers[i].neurons)
            # h = mu x next_hidden_size
            feedforward_data.append(h)
            results = self._activation_function.evaluate(h)

        return feedforward_output, feedforward_data, results

    def backpropagation(self, feedforward_output, feedforward_data, error, epoch):
        delta_W = []

        derivatives = self._activation_function.d_evaluate(feedforward_data[-1])  # mu * output_size
        delta_i = error * derivatives  # mu * output_size, elemento a elemento

        # #delta_i = mu * output_size
        # #feedforward_output[-1] = #hidden_data = mu * (hidden_size + 1)
        delta_W.append(
            self._optimization_method.adjust(delta_i, feedforward_output[-1], len(feedforward_output) - 1, epoch))
        # #delta_W =  (#hidden_size + 1) * #output_size

        for i in reversed(range(len(self._layers) - 1)):
            # delta_w tiene que tener la suma de todos los delta_w para cada iteracion para ese peso
            #        mu * output_size  *   ((hidden_size + 1 {bias_layer} - 1) * output_size).T
            error = np.dot(delta_i, np.delete(self._layers[i + 1].neurons, 0, axis=0).T)
            # mu * (hidden_size + 1 {bias_layer} - 1)  == mu * hidden_size

            # Call _optimization_method #
            derivatives = self._activation_function.d_evaluate(feedforward_data[i + 1])  # mu * hidden_size
            delta_i = error * derivatives  # mu * hidden_size
            # #feedforward[i] = mu * (previous_hidden_size + 1) ; delta_i = mu * hidden_size
            delta_W.append(self._optimization_method.adjust(delta_i, feedforward_output[i], i, epoch))
            # Me libero del mu (estoy "sumando" todos los delta_w)

        # Calculo w = w + dw

        for i in range(len(self._layers)):
            self._layers[i].neurons = np.add(self._layers[i].neurons, delta_W[-(i + 1)])

    def train_batch(self, data: ndarray[float], expected: ndarray[float]) -> list[ndarray[float]]:
        # #initial_data = mu x initial_size, #expected = mu x output_size
        error_history = []
        for epoch in tqdm(range(self._epochs)):
            feedforward_output, feedforward_data, results = self.feedforward(data)

            error = expected - results  # mu * output_size
            # ver calculo del error con llamando a d_error #
            error_history.append(mse(error))
            if self._cut_condition.is_finished(error):
                break

            self.backpropagation(feedforward_output, feedforward_data, error, epoch)

        return error_history

    def save(self, file_name: str):
        with open(file_name, "wb") as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(file_name: str):
        with open(file_name, "rb") as infile:
            return pickle.load(infile)
