import numpy as np
from ._eval_fun import eval_fun
from ._cycling import cycling
from ._get_exitflag import get_exitflag
import pdb

def inner_direct_search(fun, xbase, fbase, D, direction_indices, alpha, options):
    """
    Performs a single iteration of classical direct search within a given block.

    Args:
        fun: Objective function to minimize.
        xbase: Current point in the search space.
        fbase: Function value at xbase.
        D: Directions for polling.
        direction_indices: Indices of directions in D.
        alpha: Step size.
        options: Dictionary containing various options.

    Returns:
        xopt: Optimal point found.
        fopt: Optimal function value found.
        exitflag: Exit flag indicating the reason for termination.
        output: Dictionary containing function evaluation history and other details.
    """

    # Set the value of sufficient decrease factor.
    reduction_factor = options['reduction_factor']

    # Set target of objective function.
    ftarget = options['ftarget']

    # Set polling strategy employed within one block.
    polling_inner = options['polling_inner']

    # Set cycling strategy inside each block.
    cycling_strategy = options['cycling_inner']

    # Boolean value of WITH_CYCLING_MEMORY.
    with_cycling_memory = options['with_cycling_memory']

    # Set the forcing function.
    forcing_function = options['forcing_function']

    # Max function evaluations allowed.
    MaxFunctionEvaluations = options['MaxFunctionEvaluations']

    # Function evaluations used in this function.
    FunctionEvaluations_exhausted = options['FunctionEvaluations_exhausted']

    # Verbose output flag.
    verbose = options['verbose']

    # Initialize exit flag.
    exitflag = np.nan

    # Initialize parameters.
    n = len(xbase)
    num_directions = len(direction_indices)
    fhist = np.full(num_directions, np.nan)
    xhist = np.full((n, num_directions), np.nan)
    nf = 0
    fopt = fbase
    xopt = xbase

    for direction_idx in range(num_directions):
        # Evaluate the objective function for the current polling direction.
        # D[:, direction_idx] is a row vector. So we need to use slicing to get a column vector.
        xnew = xbase + alpha * D[:, [direction_idx]]
        # if xbase.ndim == 1:
        #     pdb.set_trace()
        fnew, fnew_real = eval_fun(fun, xnew)
        nf += 1

        # Record the real function value.
        fhist[nf - 1] = fnew_real
        xhist[:, [nf - 1]] = xnew

        if verbose:
            print(f"Function number {FunctionEvaluations_exhausted + nf}, F = {fnew_real}")
            print("The corresponding X is:")
            print(" ".join(f"{xi:.6f}" for xi in xnew.ravel()))

        # Update the best point and the best function value.
        if fnew < fopt:
            xopt = xnew
            fopt = fnew

        # Check whether the sufficient decrease condition is achieved.
        sufficient_decrease = (fnew + reduction_factor[2] * forcing_function(alpha) / 2 < fbase)

        # Opportunistic case: if sufficient decrease is achieved, stop computations.
        if sufficient_decrease and polling_inner.lower() != "complete":
            direction_indices = cycling(direction_indices, direction_idx, cycling_strategy, with_cycling_memory)
            break

        if nf >= MaxFunctionEvaluations or fnew <= ftarget:
            break

    # Determine if the algorithm terminates by the first two cases.
    terminate = (nf >= MaxFunctionEvaluations or fnew <= ftarget)
    if fnew <= ftarget:
        exitflag = get_exitflag("FTARGET_REACHED")
    elif nf >= MaxFunctionEvaluations:
        exitflag = get_exitflag("MAXFUN_REACHED")

    # Truncate fhist and xhist to the length of nf.
    output = {
        'fhist': fhist[:nf],
        'xhist': xhist[:, :nf],
        'nf': nf,
        'direction_indices': direction_indices,
        'terminate': terminate
    }

    # if xopt.ndim == 1:
    #     pdb.set_trace()

    return xopt, fopt, exitflag, output

# Note: Make sure to define eval_fun and get_exitflag functions as per your needs.