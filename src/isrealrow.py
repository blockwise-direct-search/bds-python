import numpy as np


def isrealrow(x):
    if np.size(x) == 0:
        isrr = True
        length = 0
    elif np.isrealobj(x) and np.ndim(x) == 1 and x.shape[0] == 1:
        isrr = True
        length = len(x)
    else:
        isrr = False
        length = np.nan

        return isrr, length
