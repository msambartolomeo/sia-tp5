import input
import utils
from src.plot import plot_latent
from src.variational_autoencoder import VariationalAutoencoder

LATENT_DIMENSION = 2

INPUT = input.parse_font("data/font.h")
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    settings = utils.get_settings()
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)
    encoder_architecture = utils.get_architecture(settings)
    decoder_architecture = list(reversed(encoder_architecture))

    vae = VariationalAutoencoder(
        INPUT_SIZE,
        LATENT_DIMENSION,
        epochs,
        encoder_architecture,
        decoder_architecture,
        activation_method,
        optimization_method
    )

    print(f"Training finished in {len(vae.train(INPUT))} epochs.")

    plot_latent(vae)


if __name__ == "__main__":
    main()
