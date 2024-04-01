import numpy as np


# ISREALCOLUMN checks whether x is a real column. If yes, it returns len = length(x); otherwise, len = NaN.
def isrealcolumn(x):
    if np.size(x) == 0:
        isrc = True
        length = 0
    elif (x.size == 1 and x.ndim == 1) or (np.ndim(x) == 2 and x.shape[1] == 1):
        isrc = True
        length = len(x)
    else:
        isrc = False
        length = np.nan

    return isrc, length
