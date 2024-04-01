import numpy as np
import _isrealrow
import _isrealcolumn
import pdb


def isrealvector(x):
    if _isrealrow.isrealrow(x)[0] or _isrealcolumn.isrealcolumn(x)[0]:
        isrv = True
        length = len(x)
    else:
        isrv = False
        length = np.nan
    return isrv, length