import numpy as np
from _eval_fun import eval_fun
from _cycling import cycling
from _get_exitflag import get_exitflag
import pdb


def inner_direct_search(fun, xbase, fbase, D, direction_indices, alpha, options):
    reduction_factor = options["reduction_factor"]

    ftarget = options["ftarget"]

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
        pdb.set_trace()
        # Without reshape(-1, 1), D[:, direction_indices[j - 1]] will be a 1D array, not column vector.
        xnew = xbase + alpha * D[:, direction_indices[j - 1]].reshape(-1, 1)
        [fnew, fnew_real] = eval_fun(fun, xnew)
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

        if nf >= MaxFunctionEvaluations | fnew < ftarget:
            break

    terminate = (nf >= MaxFunctionEvaluations) or (fnew <= ftarget)
    if fnew <= ftarget:
        exitflag = get_exitflag("FTARGET_REACHED", debug_flag)
    elif nf >= MaxFunctionEvaluations:
        exitflag = get_exitflag("MAXFUN_REACHED", debug_flag)

    output = {"fhist": fhist[1:nf], "xhist": xhist[:, :nf], "nf": nf, "direction_indices": direction_indices,
              "terminate": terminate}

    return [xopt, fopt, exitflag, output]
