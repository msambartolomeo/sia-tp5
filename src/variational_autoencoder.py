import copy

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import OptimizationMethod


class VariationalAutoencoder:
    def __init__(self, input_size: int, latent_size: int, epochs: int,
                 encoder_architecture: list[int],
                 decoder_architecture: list[int],
                 cut_condition: CutCondition,
                 activation_method: ActivationMethod,
                 optimization_method: OptimizationMethod):
        self._epochs = epochs

        encoder_architecture.insert(0, input_size)
        encoder_architecture.append(2 * latent_size)
        self._encoder = MultiLayerPerceptron(encoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))

        decoder_architecture.insert(0, latent_size)
        decoder_architecture.append(input_size)
        self._decoder = MultiLayerPerceptron(decoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))
