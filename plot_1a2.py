import numpy as np

import utils
from data.font import FONT
from src.activation_method import TangentActivationFunction
from src.autoencoder import Autoencoder
from src.cut_condition import OneWrongPixelCutCondition, FalseCutCondition
from src.optimization_method import AdamOptimization
from src.plot import plot_errors
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
        "name": "Tangent, Adam, []",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [],
    },
]
# Shared by all experiments
CUT_CONDITION = FalseCutCondition()
EPOCHS = 10000
REPETITIONS = 5

# Relative to the problem
LATENT_DIMENSION = 2
INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    experiment_mean = []
    experiment_std = []

    for configuration in CONFIGURATIONS:
        experiment_errors = []
        for _ in range(REPETITIONS):
            activation_method = configuration["activation_method"]
            optimization_method = configuration["optimization_method"]
            inner_architecture = configuration["inner_architecture"]
            architecture = [INPUT_SIZE] + inner_architecture + [LATENT_DIMENSION] + list(reversed(inner_architecture)) + [
                INPUT_SIZE]

            ae = Autoencoder(architecture,
                             EPOCHS,
                             CUT_CONDITION,
                             activation_method,
                             optimization_method)
            error_history = ae.train_batch(INPUT, INPUT)
            experiment_errors.append(error_history)
        curr_mean = []
        curr_std = []
        for epoch in range(EPOCHS):
            curr_mean.append(np.mean([experiment_errors[exp][epoch] for exp in range(REPETITIONS)]))
            curr_std.append(np.std([experiment_errors[exp][epoch] for exp in range(REPETITIONS)]))
        experiment_mean.append(curr_mean)
        experiment_std.append(curr_std)

    plot_errors(experiment_mean, experiment_std, [conf["name"] for conf in CONFIGURATIONS], "Configurations for autoencoder: Avg. of 5, full epoch count", "epoch", "MSE(e)")


if __name__ == "__main__":
    main()
