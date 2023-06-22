import copy

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from src.activation_method import ActivationMethod
from src.cut_condition import FalseCutCondition
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import OptimizationMethod


def loss_function(mean, std, data, result):
    rec = 0.5 * np.mean((data - result) ** 2)
    kl = -0.5 * np.sum(1 + std - mean ** 2 - np.exp(std))

    return rec + kl


def reparametrization_trick(mean: ndarray[float], std: ndarray[float]) -> tuple[ndarray[float], float]:
    eps = np.random.standard_normal()
    return eps * std + mean, eps


class VariationalAutoencoder:
    def __init__(self, input_size: int, latent_size: int, epochs: int,
                 encoder_architecture: list[int],
                 decoder_architecture: list[int],
                 activation_method: ActivationMethod,
                 optimization_method: OptimizationMethod):
        self._epochs = epochs

        self._input_size = input_size
        self._latent_size = latent_size
        self._last_delta_size = decoder_architecture[0]

        cut_condition = FalseCutCondition()

        encoder_architecture.insert(0, input_size)
        encoder_architecture.append(2 * latent_size)
        self._encoder = MultiLayerPerceptron(encoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))

        decoder_architecture.insert(0, latent_size)
        decoder_architecture.append(input_size)
        self._decoder = MultiLayerPerceptron(decoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))

    def train(self, data: ndarray[float]) -> list[float]:
        assert data.shape[1] == self._input_size

        loss_history = []
        for epoch in tqdm(range(self._epochs)):
            # NOTE: Feedforward
            result = self._encoder.feedforward(data)

            mean = result[:, :result.shape[1] // 2]
            std = result[:, result.shape[1] // 2:]

            z, eps = reparametrization_trick(mean, std)

            result = self._decoder.feedforward(z)

            loss = loss_function(mean, std, data, result)
            loss_history.append(loss)
            if loss < 0.01:
                break

            # NOTE: Decoder Backpropagation for reconstruction
            dL_dX = data - result
            decoder_gradients, last_delta = self._decoder.backpropagation(dL_dX)

            # NOTE: Encoder backpropagation for reconstruction
            dz_dm = np.ones([self._last_delta_size, self._latent_size])
            dz_dv = eps * np.ones([self._last_delta_size, self._latent_size])
            mean_error = np.dot(last_delta, dz_dm)
            std_error = np.dot(last_delta, dz_dv)
            encoder_reconstruction_error = np.concatenate((mean_error, std_error), axis=1)
            encoder_reconstruction_gradients, _ = self._encoder.backpropagation(encoder_reconstruction_error)

            # NOTE: Encoder backpropagation for regularization
            dL_dm = mean
            dL_dv = 0.5 * (np.exp(std) - 1)
            encoder_loss_error = np.concatenate((dL_dm, dL_dv), axis=1)
            encoder_loss_gradients, _ = self._encoder.backpropagation(encoder_loss_error)

            # NOTE: update weights with gradients
            encoder_gradients = []
            for g1, g2 in zip(encoder_loss_gradients, encoder_reconstruction_gradients):
                encoder_gradients.append(g1 + g2)

            self._encoder.update_weights(encoder_gradients, epoch)
            self._decoder.update_weights(decoder_gradients, epoch)

        return loss_history

    def predict(self, z: ndarray[float]) -> ndarray[float]:
        assert z.shape[1] == self._latent_size

        return self._decoder.feedforward(z)
