import numpy as np
from numpy import arange

from data.font import FONT, FONT_LETTERS
from src.activation_method import StepActivationFunction
from src.autoencoder import Autoencoder
from src.plot import plot_scatter, plot_letter_pattern, plot_latent_grid

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
MIN_X, MAX_X, MIN_Y, MAX_Y = -1.5, 1.5, -1.5, 1.5
COUNT = 20


def main():
    ae = Autoencoder.load("data/e1a.mlp")
    latent_space = []
    stepified_latent_space = []
    for y in arange(MAX_Y, MIN_Y, -(MAX_Y - MIN_Y) / COUNT):
        for x in arange(MIN_X, MAX_X, (MAX_X - MIN_X) / COUNT):
            pattern = ae.create_from_latent(np.array([x, y]))[0]
            latent_space.append(pattern)
            pattern = StepActivationFunction().evaluate(pattern)
            stepified_latent_space.append(pattern)
    print(latent_space)
    plot_latent_grid(np.array(latent_space), COUNT, COUNT)
    plot_latent_grid(np.array(stepified_latent_space), COUNT, COUNT)


if __name__ == "__main__":
    main()
