import numpy as np


# ISREALCOLUMN checks whether x is a real column. If yes, it returns len = length(x); otherwise, len = NaN.
def isrealcolumn(x):
    if np.size(x) == 0:
        isrc = True
        length = 0
    elif np.isrealobj(x) and np.ndim(x) == 1 and x.shape[0] == 1:
        isrc = True
        length = len(x)
    else:
        isrc = False
        length = np.nan

    return isrc, length
