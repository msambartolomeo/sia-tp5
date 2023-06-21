import numpy as np


def parse_font(path: str):
    font = []
    with open(path, "r") as f:
        for i in range(5):
            f.readline()

        for i in range(32):
            letter = []
            line = f.readline()
            line = line.split("}")[0]
            hexs = line.split(",")
            for j in range(7):
                l = int(hexs[j].strip("{ }"), 16)

                for k in reversed(range(5)):
                    pixel = (l >> k) & 1
                    pixel = -1 if pixel == 0 else 1

                    letter.append(pixel)

            font.append(letter)

    return np.array(font)
