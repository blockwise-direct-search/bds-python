import numpy as np
from _isrealrow import isrealrow
from _isrealcolumn import isrealcolumn
import pdb


def isrealvector(x):
    if isrealrow(x)[0] or isrealcolumn(x)[0]:
        isrv = True
        length = len(x)
    else:
        isrv = False
        length = np.nan
    return isrv, length