import numpy as np

def is_row_vector(value):
    r"""
    Verify if the input variable is a row vector (including list and NumPy array)
    """
    if isinstance(value, list):  # Check if it is a list
        return all(isinstance(x, (int, float)) for x in value)  # Make sure all 
    # elements are integers or floats
    
    # Check if it is a NumPy array
    return isinstance(value, np.ndarray) and \
        (value.ndim == 1 or (value.ndim == 2 and value.shape[0] == 1))