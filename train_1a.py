import math

import numpy as np

import utils
from data.font import FONT
from src.autoencoder import Autoencoder
from src.multi_layer_perceptron import MultiLayerPerceptron

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1]  # 35


def main():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    ae = Autoencoder([INPUT_SIZE, 25, 25, 25, 25,
                                            2,
                                            25, 25, 25, 25, INPUT_SIZE],
                               epochs,
                               cut_condition,
                               activation_method,
                               optimization_method)
    print(f"Training finished in {len(ae.train_batch(INPUT, INPUT))} epochs.")

    ans = ae.predict(INPUT)
    for i in range(len(ans)):
        print(f"{INPUT[i]}\n{ans[i]}\n----------------------------------------------------\n")
        ans[i] = [1 if num >= 0 else -1 for num in ans[i]]

    for test in range(INPUT.shape[0]):
        count = 0
        for i in range(len(ans[test])):
            if INPUT[test][i] != ans[test][i]:
                count += 1

        print(f" - {count}")

    ae.save("data/e1a.mlp")


if __name__ == "__main__":
    main()
