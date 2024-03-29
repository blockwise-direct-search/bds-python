import numpy as np
import isrealrow
import isrealcolumn
import pdb


def isrealvector(x):
    if isrealrow.isrealrow(x)[0] or isrealcolumn.isrealcolumn(x)[0]:
        isrv = True
        length = len(x)
    else:
        isrv = False
        length = np.nan
    return isrv, length
