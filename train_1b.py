import numpy as np

import utils
from data.font import FONT
from src.activation_method import StepActivationFunction
from src.autoencoder import Autoencoder

LATENT_DIMENSION = 2

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)
    inner_architecture = utils.get_architecture(settings)
    architecture = [INPUT_SIZE] + inner_architecture + [LATENT_DIMENSION] + list(reversed(inner_architecture)) + [
        INPUT_SIZE]
    noise = utils.get_noise(settings)
    NOISY_INPUT = np.array([utils.add_noise(letter, noise) for letter in INPUT])
    NOISY_INPUT_TEST = np.array([utils.add_noise(letter, noise) for letter in INPUT])

    ae = Autoencoder(architecture,
                     epochs,
                     cut_condition,
                     activation_method,
                     optimization_method)
    print(f"Training finished in {len(ae.train_batch(NOISY_INPUT, INPUT))} epochs.")

    ans = ae.predict(NOISY_INPUT_TEST)
    ans = StepActivationFunction().evaluate(ans)

    for test in range(INPUT.shape[0]):
        count = 0
        for i in range(len(ans[test])):
            if INPUT[test][i] != ans[test][i]:
                count += 1

        print(f" - {count}")

    ae.save("data/e1b.mlp")


if __name__ == "__main__":
    main()
