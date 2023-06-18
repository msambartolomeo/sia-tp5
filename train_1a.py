import numpy as np

import utils
from src.multi_layer_perceptron import MultiLayerPerceptron
from data.font import FONT

INPUT = np.array([[pixel for line in letter for pixel in line] for letter in FONT])
INPUT_SIZE = INPUT.shape[1] # 35


def main():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    mlp = MultiLayerPerceptron([INPUT_SIZE, 2, INPUT_SIZE],
                               epochs,
                               cut_condition,
                               activation_method,
                               optimization_method)
    print(f"Training finished in {len(mlp.train_batch(INPUT, INPUT))} epochs.")

    ans = mlp.predict(INPUT)
    for test in range(INPUT.shape[0]):
        print(f"{INPUT[test]}\n{ans[test]}\n----------------------------------------------------\n")

    mlp.save("data/e1a.mlp")


if __name__ == "__main__":
    main()
