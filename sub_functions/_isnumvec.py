import numpy as np

def isnumvec(value):
    r"""
    Check whether the input is a numerical vector or a single number.
    """
    if isinstance(value, (int, float)):  # Check if it is a single number
        return True
    elif isinstance(value, list):  # Check if it is a list of numbers
        return all(isinstance(x, (int, float)) for x in value)
    elif isinstance(value, np.ndarray):  # Check if it is a numpy array
        return (np.issubdtype(value.dtype, np.number) and 
                (value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1)))
    return False