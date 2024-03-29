import numpy as np


def isrealmatrix(x):
    if np.size(x) == 0:
        isrc = True
        m = 0
        n = 0
    elif np.issubdtype(x.dtype, np.number) and np.isreal(x) and x.ndim <= 2:
        isrc = True
        m, n = x.shape
    else:
        isrc = False
        m = np.nan
        n = np.nan
