import numpy as np
import sys
import os
# Get the path of the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project root directory (project directory) to sys.path.
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_dir)
from main import bds

def chrosen(x):
    """
    ROSENBROCK_EXAMPLE illustrates how to use bds.

    ***********************************************************************
    Authors:    Haitian Li (hai-tian.li@connect.polyu.hk)
                and Zaikun ZHANG (zaikun.zhang@polyu.edu.hk)
                Department of Applied Mathematics,
                The Hong Kong Polytechnic University

    ***********************************************************************
    """
    f = np.sum((x[:-1] - 1) ** 2 + 4 * (x[1:] - x[:-1] ** 2) ** 2)
    # Uncomment the line below if you want to add noise to the function value
    # f *= (1 + 1e-8 * np.random.randn())
    return f

def rosenbrock_example(options = None, x0 = None):
    # Set options to an empty dict if it is not provided.
    if options is None:
        options = {}
    # Set the default value of x0 to [0, 0, 0] if it is not provided.
    if x0 is None:
        x0 = np.array([0, 0, 0])
    options["verbose"] = True
    bds(chrosen, x0, options)

rosenbrock_example()