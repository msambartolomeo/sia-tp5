import numpy as np

import utils
from data.font import FONT
from src.activation_method import TangentActivationFunction, IdentityActivationFunction
from src.autoencoder import Autoencoder
from src.cut_condition import OneWrongPixelCutCondition, FalseCutCondition
from src.optimization_method import AdamOptimization
from src.plot_classes import MultiErrorVsEpochTestPlotter

# Configurations for the experiments
CONFIGURATIONS = [
    {
        "name": "Tangent, Adam, [25, 25, 25, 25]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [25, 25, 25, 25],
    },
    {
        "name": "Tangent, Adam, [10]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [10],
    },
    {
        "name": "Identity, Adam, [25, 25, 25]",
        "activation_method": IdentityActivationFunction(),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [25, 25, 25],
    },
]
# Shared by all experiments
CUT_CONDITION = FalseCutCondition()
EPOCHS = 100000

# Relative to the problem
LATENT_DIMENSION = 2
INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    MultiErrorVsEpochTestPlotter("e1a2_font.png",
                                 f"Different configurations: test count = 10, all to max epochs allowed",
                                 "Epoch",
                                 "Error(MSE)",
                                 "Conf",
                                 [conf["name"] for conf in CONFIGURATIONS]
                                 ).plot(
        (lambda: [
            Autoencoder([INPUT_SIZE] + conf["inner_architecture"] + [LATENT_DIMENSION] + list(reversed(conf["inner_architecture"])) + [INPUT_SIZE],
                        EPOCHS,
                        CUT_CONDITION,
                        conf["activation_method"],
                        conf["optimization_method"]
                        ).train_batch(INPUT, INPUT)

            for conf in CONFIGURATIONS]
         ))


if __name__ == "__main__":
    main()
