import os

import numpy as np
from matplotlib import pyplot as plt, ticker

GRID_WIDTH = 8
GRID_HEIGHT = 4
LETTER_WIDTH = 5
LETTER_HEIGHT = 7

OUTPUT_DIR = "plots/"

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


def plot_scatter(x, y, labels):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i in range(len(x)):
        ax.annotate(labels[i], (x[i], y[i]))
    plt.show()


def plot_latent(vae, n=20, fig_size=15, digit_size=7):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-1.0, 1.0, n)
    grid_y = np.linspace(-1.0, 1.0, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = np.array([[xi, yi]])
            output = vae.predict(z)
            digit = output[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(fig_size, fig_size))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
