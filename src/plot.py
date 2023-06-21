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


def plot_grid(data, grid=False, minor_width=3):
    fig, ax = plt.subplots()
    ax.spines[:].set_visible(False)
    if grid:
        ax.set_xticks(np.arange(data.shape[1] // LETTER_WIDTH + 1) * LETTER_WIDTH - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] // LETTER_HEIGHT + 1) * LETTER_HEIGHT - .5, minor=True)
    else:
        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="tab:gray", linestyle='-', linewidth=minor_width)
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


def plot_latent_grid(data, height, width):
    row_count = LETTER_HEIGHT * height
    col_count = LETTER_WIDTH * width
    aux = np.zeros([row_count, col_count])
    for row in range(row_count):
        for col in range(col_count):
            letter = data[col // LETTER_WIDTH + row // LETTER_HEIGHT * width]
            pixel = letter[col % LETTER_WIDTH + (row % LETTER_HEIGHT) * LETTER_WIDTH]
            aux[row][col] = pixel

    plot_grid(aux, grid=True, minor_width=1)


def plot_scatter(x, y, labels):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i in range(len(x)):
        ax.annotate(labels[i], (x[i], y[i]))
    plt.show()


def plot_errors(mean, std, labels, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    for i in range(len(mean)):
        curr_mean = np.array(mean[i])
        curr_std = np.array(std[i])
        x = np.arange(curr_mean.shape[0])

        ax.fill_between(x, curr_mean + curr_std, curr_mean - curr_std, alpha=.5, linewidth=0)
        ax.plot(x, curr_mean, linewidth=2, label=labels[i])

    leg = plt.legend(loc='upper right')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    current_size = fig.get_size_inches()
    new_size = (current_size[0] * 1.5, current_size[1] * 1.5)
    fig.set_size_inches(new_size)
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    plt.show()
