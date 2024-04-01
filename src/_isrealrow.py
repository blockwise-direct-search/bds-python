import numpy as np
import pdb

def isrealrow(x):
    if np.size(x) == 0:
        isrr = True
        length = 0
    elif (x.size == 1 and x.ndim == 1) or np.ndim(x) == 1:
        isrr = True
        length = len(x)
    else:
        isrr = False
        length = np.nan
    return isrr, length
