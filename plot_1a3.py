import numpy as np

from data.font import FONT
from src.autoencoder import Autoencoder
from src.plot import plot_scatter

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])


def main():
    ae = Autoencoder.load("data/e1a.mlp")
    latent_values = ae.get_latent(INPUT)
    plot_scatter(latent_values[:][0], latent_values[:][1])


if __name__ == "__main__":
    main()
