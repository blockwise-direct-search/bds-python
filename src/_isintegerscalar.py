import numpy as np
import _isrealscalar


# ISINTEGERSCALAR checks whether x is an integer scalar.
def isintegerscalar(x):
    isis = _isrealscalar.isrealscalar(x) and (x % 1 == 0)
    return isis
