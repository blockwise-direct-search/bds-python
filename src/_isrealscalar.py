import numpy as np

# ISREALSCALAR checks whether x is a real scalar.
def isrealscalar(x):
    isrs = np.isscalar(x) and np.isreal(x) and np.ndim(x) == 0
    return isrs
