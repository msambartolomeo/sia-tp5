import numpy as np

from data.font import FONT
from src.autoencoder import Autoencoder
from src.plot import plot_letter_pattern, plot_font

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])


def main():
    plot_font(INPUT)
    ae = Autoencoder.load("data/e1a.mlp")
    ans = ae.predict(INPUT)
    print(ans.shape)
    plot_font(ans)


if __name__ == "__main__":
    main()
