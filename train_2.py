import numpy as np

import utils
from data.font import FONT
from src.variational_autoencoder import VariationalAutoencoder

LATENT_DIMENSION = 2

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    settings = utils.get_settings()
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)
    encoder_architecture = utils.get_architecture(settings)
    decoder_architecture = list(reversed(encoder_architecture))

    ae = VariationalAutoencoder(
        INPUT_SIZE,
        LATENT_DIMENSION,
        epochs,
        encoder_architecture,
        decoder_architecture,
        activation_method,
        optimization_method
    )

    print(f"Training finished in {len(ae.train(INPUT))} epochs.")
    #
    # ans = ae.predict(INPUT)
    # for i in range(len(ans)):
    #     print(f"{INPUT[i]}\n{ans[i]}\n----------------------------------------------------\n")
    #     ans[i] = [1 if num >= 0 else -1 for num in ans[i]]
    #
    # for test in range(INPUT.shape[0]):
    #     count = 0
    #     for i in range(len(ans[test])):
    #         if INPUT[test][i] != ans[test][i]:
    #             count += 1
    #
    #     print(f" - {count}")
    #
    # ae.save("data/e1a.mlp")


if __name__ == "__main__":
    main()
