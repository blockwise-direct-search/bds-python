import numpy as np
import _eval_fun
from _cycling import cycling


def inner_direct_search(fun, xbase, fbase, D, direction_indices, alpha, options):
    reduction_factor = options["reduction_factor"]

    polling_inner = options["polling_inner"]

    cycling_strategy = options["cycling_inner"]

    with_cycling_memory = options["with_cycling_memory"]

    forcing_function = options["forcing_function"]

    MaxFunctionEvaluations = options["MaxFunctionEvaluations"]

    FunctionEvaluations_exhausted = options["FunctionEvaluations_exhausted"]

    iprint = options["iprint"]

    debug_flag = options["debug_flag"]

    exitflag = np.nan

    n = len(xbase)
    num_directions = len(direction_indices)
    fhist = np.full((1, MaxFunctionEvaluations), np.nan)
    xhist = np.full((n, MaxFunctionEvaluations), np.nan)
    nf = 0
    fopt = fbase
    xopt = xbase

    for j in range(1, num_directions + 1):

        xnew = xbase + alpha * D[:, direction_indices[j - 1]]
        [fnew, fnew_real] = _eval_fun.eval_fun(fun, xnew)
        nf = nf + 1
        fhist = np.append(fhist, fnew_real)
        xhist = np.append(xhist, xnew)
        if iprint > 0:
            print("Function number {}, F = {:.6f}".format(FunctionEvaluations_exhausted + nf, fnew))
            print("The corresponding X is:")
            print(" ".join("{:.6f}".format(x) for x in xnew.flatten()))
            print()

        if fnew < fopt:
            xopt = xnew
            fopt = fnew

        sufficient_decrease = (fnew + reduction_factor(3) * forcing_function(alpha) / 2 < fbase)

        if sufficient_decrease and polling_inner.lower() == "complete":
            direction_indices = cycling(direction_indices, j, cycling_strategy, with_cycling_memory, debug_flag)
            break

        
