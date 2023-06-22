import numpy as np
from tqdm import tqdm

import utils
from data.font import FONT
from src.activation_method import TangentActivationFunction, StepActivationFunction
from src.autoencoder import Autoencoder
from src.cut_condition import FalseCutCondition
from src.optimization_method import AdamOptimization, MomentumOptimization
from src.plot import plot_errors

# Configurations for the experiments
CONFIGURATIONS = [
    {
        "name": "Adam, [25, 25, 25, 25]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [25, 25, 25, 25],
    },
    {
        "name": "Momentum, [25, 25, 25, 25]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": MomentumOptimization(),
        "inner_architecture": [25, 25, 25, 25],
    },
    {
        "name": "Adam, [25, 25, 25]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [25, 25, 25],
    },
    {
        "name": "Adam, [10, 10]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [10, 10],
    },
    {
        "name": "Adam, [18]",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [18],
    },
    {
        "name": "Adam, []",
        "activation_method": TangentActivationFunction(0.5),
        "optimization_method": AdamOptimization(),
        "inner_architecture": [],
    },
]
# Shared by all experiments
CUT_CONDITION = FalseCutCondition()
EPOCHS = 10000
REPETITIONS = 10

# Relative to the problem
LATENT_DIMENSION = 4
INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    experiments_mean = []
    experiments_std = []

    for configuration in tqdm(CONFIGURATIONS):
        experiment_mean = []
        experiment_std = []
        for noise in np.arange(0, 1, 0.1):
            NOISY_INPUT = utils.add_noise(INPUT, noise)
            NOISY_INPUT_TEST = utils.add_noise(INPUT, noise)
            errors_of_same_noise = []
            for _ in range(REPETITIONS):
                activation_method = configuration["activation_method"]
                optimization_method = configuration["optimization_method"]
                inner_architecture = configuration["inner_architecture"]
                architecture = [INPUT_SIZE] + inner_architecture + [LATENT_DIMENSION] + list(
                    reversed(inner_architecture)) + [
                                   INPUT_SIZE]

                ae = Autoencoder(architecture,
                                 EPOCHS,
                                 CUT_CONDITION,
                                 activation_method,
                                 optimization_method)
                ae.train_batch(NOISY_INPUT, INPUT)
                prediction = ae.predict(NOISY_INPUT_TEST)
                prediction = StepActivationFunction().evaluate(prediction)
                max_error = max([utils.count_different_pixels(prediction[letter], INPUT[letter]) for letter in range(prediction.shape[0])])
                errors_of_same_noise.append(max_error)
            experiment_mean.append(np.mean(errors_of_same_noise))
            experiment_std.append(np.std(errors_of_same_noise))
        experiments_mean.append(experiment_mean)
        experiments_std.append(experiment_std)

    plot_errors(experiments_mean, experiments_std, [conf["name"] for conf in CONFIGURATIONS],
                "Denoising autoencoders: Avg. of 5, full epoch count", "noise", "Max. pixel difference", x=np.arange(0, 1, 0.1))


if __name__ == "__main__":
    main()
