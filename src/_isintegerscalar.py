import numpy as np
from _isrealscalar import isrealscalar


# ISINTEGERSCALAR checks whether x is an integer scalar.
def isintegerscalar(x):
    isis = isrealscalar(x) and (x % 1 == 0)
    return isis
