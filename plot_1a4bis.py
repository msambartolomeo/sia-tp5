import numpy as np

from data.font import FONT
from src.activation_method import StepActivationFunction
from src.autoencoder import Autoencoder
from src.plot import plot_letter_pattern, plot_font

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])


def main():
    ae = Autoencoder.load("data/e1a.mlp")
    LATENT_SPACE_POINTS = [
        (0, 0),
        (-.6, 0),
        (.6, -.6),
        (1, 1),
    ]
    for x, y in LATENT_SPACE_POINTS:
        pattern = ae.create_from_latent(np.array([x, y]))[0]
        plot_letter_pattern(pattern)
        pattern = StepActivationFunction().evaluate(pattern)
        plot_letter_pattern(pattern)


if __name__ == "__main__":
    main()
