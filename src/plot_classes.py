import os.path
import statistics
from abc import ABC
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import utils
from src.activation_method import StepActivationFunction, IdentityActivationFunction, TangentActivationFunction, \
    SigmoidActivationFunction, LogisticActivationFunction
from src.cut_condition import FalseCutCondition
from src.error import mse
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import GradientDescentOptimization, MomentumOptimization


OUTPUT_DIR = "figs/"
TEST_COUNT = 100
MAX_EPOCHS = 1000
LEARNING_RATE = 0.01


class TestPlotter(ABC):

    def __init__(self, out_name: str):
        self._out_name = out_name

    def plot(self, test: Callable):
        data = self._create_data()

        for t in range(TEST_COUNT):
            self._add_data(data, test())

        data = self._post_process(data)

        self._save_plot(data)

    def _create_data(self):
        raise NotImplementedError()

    def _add_data(self, data, new_data):
        raise NotImplementedError

    def _post_process(self, data):
        raise NotImplementedError()

    def _save_plot(self, data):
        raise NotImplementedError()


class ErrorVsEpochTestPlotter(TestPlotter):
    def __init__(self, out_name, title, xaxis, yaxis):
        super().__init__(out_name)
        self._title = title
        self._xaxis = xaxis
        self._yaxis = yaxis

    def _create_data(self):
        return []

    def _add_data(self, data, new_data):
        data.append(new_data)

    def _post_process(self, data):
        max_size = max([len(history) for history in data])
        mean, std = [], []
        for i in range(max_size):
            data_of_current_epoch = list(filter(
                lambda x: x is not None,
                [history[i] if i < len(history) else None for history in data]
            ))
            mean.append(statistics.mean(data_of_current_epoch))
            if len(data_of_current_epoch) >= 2:
                std.append(statistics.stdev(data_of_current_epoch))
            else:
                std.append(.0)

        return {
            "mean": mean,
            "std": std,
        }

    def _save_plot(self, data):
        # plot
        fig, ax = plt.subplots()
        self._plot_line(fig, ax, data)

        plt.title(self._title, ontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()

        plt.savefig(OUTPUT_DIR + self._out_name)

    def _plot_line(self, fig, ax, data, label=None):
        mean = np.array(data["mean"])
        std = np.array(data["std"])

        x = np.arange(mean.shape[0])

        ax.fill_between(x, mean + std, mean - std, alpha=.5, linewidth=0, label=label)
        ax.plot(x, mean, linewidth=2)


class ErrorVsTrainRatioTestPlotter:
    def __init__(self, out_name, title, xaxis, yaxis):
        self._out_name = out_name
        self._title = title
        self._xaxis = xaxis
        self._yaxis = yaxis

    def plot(self, test: Callable):
        data = []

        for i, ratio in enumerate(np.arange(0, 1, 0.05)):
            data.append([])
            for t in range(TEST_COUNT):
                data[i].append(test(ratio))

        data = self._post_process(data)

        self._save_plot(data)

    def _post_process(self, data):
        mean, std = [], []
        for data_of_current_ratio in data:
            mean.append(statistics.mean(data_of_current_ratio))
            if len(data_of_current_ratio) >= 2:
                std.append(statistics.stdev(data_of_current_ratio))
            else:
                std.append(.0)

        return {
            "mean": mean,
            "std": std,
        }

    def _save_plot(self, data):
        # plot
        fig, ax = plt.subplots()
        self._plot_line(fig, ax, data)

        plt.title(self._title)
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()

        plt.savefig(OUTPUT_DIR + self._out_name)

    def _plot_line(self, fig, ax, data, label=None):
        mean = np.array(data["mean"])
        std = np.array(data["std"])

        x = np.arange(0, 1, 0.05)

        ax.fill_between(x, mean + std, mean - std, alpha=.5, linewidth=0, label=label)
        ax.plot(x, mean, 'o-', linewidth=2)


class MultiErrorVsEpochTestPlotter(ErrorVsEpochTestPlotter):
    def __init__(self, out_name, title, xaxis, yaxis, label_type, labels):
        super().__init__(out_name, title, xaxis, yaxis)
        self._label_type = label_type
        self._labels = labels

    def _create_data(self):
        data = {}
        for label in self._labels:
            data[label] = []
        return data

    def _add_data(self, data, new_data):
        for i, label in enumerate(self._labels):
            data[label].append(new_data[i])

    def _post_process(self, data):
        post_data = {}
        for label in self._labels:
            post_data[label] = super()._post_process(data[label])

        return post_data

    def _save_plot(self, data):
        fig, ax = plt.subplots()
        for label in self._labels:
            super()._plot_line(fig, ax, data[label], f"{self._label_type} = {label}")

        plt.title(self._title)
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()
        leg = plt.legend(loc='upper right')

        plt.savefig(OUTPUT_DIR + self._out_name)
