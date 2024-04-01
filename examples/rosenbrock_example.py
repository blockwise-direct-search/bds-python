import numpy as np


def chrosen(x):
    n = len(x)
    f = np.sum((x[:n - 1] - 1) ** 2 + 4 * (x[1:n] - x[:n - 1] ** 2) ** 2)
    return f
