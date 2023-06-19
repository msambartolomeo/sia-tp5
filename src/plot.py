import os

import numpy as np
from matplotlib import pyplot as plt, ticker

GRID_WIDTH = 8
GRID_HEIGHT = 4
LETTER_WIDTH = 5
LETTER_HEIGHT = 7

OUTPUT_DIR = "../plots/"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def plot_grid(data, grid = False):
    fig, ax = plt.subplots()
    ax.spines[:].set_visible(False)
    if grid:
        ax.set_xticks(np.arange(data.shape[1] // LETTER_WIDTH + 1) * LETTER_WIDTH - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] // LETTER_HEIGHT + 1) * LETTER_HEIGHT - .5, minor=True)
    else:
        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="tab:gray", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(left=False, bottom=False)
    plt.style.use('grayscale')
    plt.imshow(data)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    plt.show()


def plot_letter_pattern(data):
    data = data.reshape(7, 5)
    plot_grid(data)


def plot_font(data):
    # data -> [32, 35]
    row_count = LETTER_HEIGHT * GRID_HEIGHT
    col_count = LETTER_WIDTH * GRID_WIDTH
    aux = np.zeros([row_count, col_count])
    for row in range(row_count):
        for col in range(col_count):
            letter = data[col // LETTER_WIDTH + row // LETTER_HEIGHT * GRID_WIDTH]
            pixel = letter[col % LETTER_WIDTH + (row % LETTER_HEIGHT) * LETTER_WIDTH]
            aux[row][col] = pixel

    plot_grid(aux, grid=True)
