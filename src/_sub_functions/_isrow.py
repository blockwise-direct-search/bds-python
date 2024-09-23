import numpy as np

def isrow(value):
    r"""
    Verify if the input variable is a row vector (including list and NumPy array)
    """
    if isinstance(value, list):  # Check if it is a list
        # Make sure all elements are integers or floats
        return all(isinstance(x, (int, float)) for x in value)
    else:
        # Check if it is a NumPy array
        return isinstance(value, np.ndarray) and (value.ndim == 1)