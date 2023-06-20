import copy

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import OptimizationMethod


class VariationalAutoencoder:
    def __init__(self, encoder_architecture: list[int], decoder_architecture: list[int], cut_condition: CutCondition,
                 activation_method: ActivationMethod, optimization_method: OptimizationMethod, epochs: int):
        self._epochs = epochs

        self._encoder = MultiLayerPerceptron(encoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))

        self._decoder = MultiLayerPerceptron(decoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))
