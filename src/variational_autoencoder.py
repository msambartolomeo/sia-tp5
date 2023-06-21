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
        loss_history = []
        for epoch in tqdm(range(self._epochs)):
            # NOTE: Feedforward
            result = self._encoder.feedforward(data)

            mean = result[:len(result) // 2]
            std = result[len(result) // 2:]

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
            dz_dm = 1
            dz_dv = eps
            encoder_reconstruction_error = np.concatenate(last_delta * dz_dm, last_delta * dz_dv)
            encoder_reconstruction_gradients, _ = self._encoder.backpropagation(encoder_reconstruction_error)

            # NOTE: Encoder backpropagation for regularization
            dL_dm = mean
            dL_dv = 0.5 * (np.exp(std) - 1)
            encoder_loss_error = np.concatenate(dL_dm, dL_dv)
            encoder_loss_gradients, _ = self._encoder.backpropagation(encoder_loss_error)

            # NOTE: update weights with gradients
            encoder_gradients = []
            for g1, g2 in zip(encoder_loss_gradients, encoder_reconstruction_gradients):
                encoder_gradients.append(np.sum(g1, g2))

            self._encoder.update_weights(encoder_gradients, epoch)
            self._decoder.update_weights(decoder_gradients, epoch)

        return loss_history
