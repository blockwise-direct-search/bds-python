import numpy as np
import pdb
from src import bds
pdb.set_trace()

def chrosen(x):
    """
    The objective function defined by the Rosenbrock function.
    
    Args:
        x (numpy.ndarray): Input vector.
        
    Returns:
        float: Function value.
    """
    f = np.sum((x[:-1] - 1) ** 2 + 4 * (x[1:] - x[:-1] ** 2) ** 2)
    # Uncomment the line below if you want to add noise to the function value
    # f *= (1 + 1e-8 * np.random.randn())
    return f

options = {"Algorithm": "cbds", "MaxFunctionEvaluations_dim_factor": 500}
bds(chrosen, np.array([0, 0, 0]), options)