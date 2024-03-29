import numpy as np
import isrealscalar


# ISINTEGERSCALAR checks whether x is an integer scalar.
def isintegerscalar(x):
    isis = isrealscalar.isrealscalar(x) and (x % 1 == 0)
    return isis
