import numpy as np

def isrealmatrix(value):
    r"""
    Verify whether the input variable is a matrix (two-dimensional array) or 
    a valid one-dimensional array
    """
    return isinstance(value, np.ndarray) and (value.ndim == 1 or value.ndim == 2)