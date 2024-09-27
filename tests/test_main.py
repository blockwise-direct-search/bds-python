import pytest
import sys
import os

# Get the path of the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project root directory (project directory) to sys.path.
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_dir)
from sub_functions import *
from main import bds

def test_bds():
    """
    BDS_TEST tests the bds function using the Chained Rosenbrock function.
    """

    def chrosen(x):
        """
        CHROSEN calculates the function value, gradient, and Hessian of the
        Chained Rosenbrock function.

        See:
        [1] Toint (1978), 'Some numerical results using a sparse matrix
        updating formula in unconstrained optimization'
        [2] Powell (2006), 'The NEWUOA software for unconstrained
        optimization without derivatives'
        """
        n = len(x)
        alpha = 4

        f = 0  # Function value
        g = np.zeros(n)  # Gradient
        H = np.zeros((n, n))  # Hessian

        for i in range(n - 1):
            f += (x[i] - 1) ** 2 + alpha * (x[i] ** 2 - x[i + 1]) ** 2
        return f

    x0 = np.zeros(3)

    # First call to bds
    _, fopt, _, _ = bds(chrosen, x0)
    assert np.isclose(fopt, 0), "The function value is not close to 0."

    options = {
        'verbose': False,
        'MaxFunctionEvaluations_dim_factor': 500,
        'ftarget': -np.inf,
        'output_alpha_hist': True,
        'output_xhist': True,
        'debug_flag': True
    }

    # Second call to bds
    _, fopt, _, _ = bds(chrosen, x0, options)
    assert np.isclose(fopt, 0), "The function value is not close to 0."

    # Testing different algorithms
    algorithms = ["pbds", "rbds", "pads", "scbds", "ds"]
    tolerances = [1e-8, 1e-6, 1e-6, 1e-10, 1e-6]
    # options['Algorithm'] = "pads"
    # _, fopt, _, _ = bds(chrosen, x0, options)

    for algo, tol in zip(algorithms, tolerances):
        options['Algorithm'] = algo
        _, fopt, _, _ = bds(chrosen, x0, options)
        assert abs(fopt) < tol, f'The function value for {algo} is not close to 0.'

if __name__ == '__main__':
    pytest.main([__file__])