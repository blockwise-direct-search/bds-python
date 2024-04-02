import numpy as np


def isnumvec(x):
    is_numvec = (isinstance(x, list) or isinstance(x, np.ndarray)) and (
                np.ndim(x) == 1 or (np.ndim(x) == 2 and (np.shape(x)[0] == 1 or np.shape(x)[1] == 1)))
    return is_numvec
