import numpy as np

def isrealvector(value):
    r"""
    Verify if the input is a numerical vector (row vector, column vector, or 
    scalar) and return the length of the vector.
    """
    if isinstance(value, (int, float)):  # Check if it is a single number
        return True, 1 # return the length of the scalar
    elif isinstance(value, list):  # Check if it is a list of numbers
        if all(isinstance(x, (int, float)) for x in value):  # Check if all
            # elements are numbers
            return True, len(value)  # Return the length of the list
    elif isinstance(value, np.ndarray):  # Check if it is a numpy array
        if (np.issubdtype(value.dtype, np.number) and 
            (value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1)) and 
            np.all(np.isreal(value))):  # Make sure it is a real vector
            return True, value.size  # Return the length of the vector
    return False, np.nan  # Return False if it is not a numerical vector