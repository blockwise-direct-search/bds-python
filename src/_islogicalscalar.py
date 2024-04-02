import numpy as np
from _isrealscalar import isrealscalar

# ISLOGICALSCALAR checks whether x is a logical scalar, including 0 and 1.
def islogicalscalar(x):

    if isinstance(x, bool) and np.isscalar(x):
        isis = True
    elif isrealscalar(x) and (x == 1 or x == 0):
        isis = True
    else:
        isis = False

        return isis
