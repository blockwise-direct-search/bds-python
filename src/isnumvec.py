import numpy as np

# ISNUMVEC checks whether x is a numeric vector.
def isnumvec(x):

    is_numvec = np.issubdtype(x.dtype, np.number) and x.ndim == 1
    return is_numvec
