import numpy as np
from _isrealvector import isrealvector
from _isrealscalar import isrealscalar
from _isintegerscalar import isintegerscalar
from _isrealmatrix import isrealmatrix
from _eval_fun import eval_fun

def verify_postconditions(fun, xopt, fopt, exitflag, output):
    # Verify whether xopt is a real vector.
    if not isrealvector(xopt):
        raise ValueError("xopt is not a real vector.")

    # Verify whether fopt is a real number.
    if not isrealscalar(fopt):
        raise ValueError("fopt is not a real number.")

    # Verify whether fun(xopt) == fopt.
    # For CUTest problem, function handle only accept column vector.
    if not eval_fun(fun, xopt) == fopt:
        raise ValueError("fun(xopt) is not equal to fopt.")

    # Verify whether exitflag is an integer.
    if not isintegerscalar(exitflag):
        raise ValueError("exitflag is not an integer.")

    # Verify whether nf is a positive integer.
    if "funcCount" not in output:
        raise ValueError("funcCount does not exist in output.")
    nf = output["funcCount"]
    if not (isintegerscalar(nf) and nf > 0):
        raise ValueError("funcCount is not a positive integer.")

    # Verify whether output is a dictionary.
    if not isinstance(output, dict):
        raise ValueError("output is not a dictionary.")

    # Verify whether output.fhist exists.
    if "fhist" not in output:
        raise ValueError("output.fhist does not exist.")
    fhist = output["fhist"]
    # Verify whether fopt is the minimum of fhist.
    if not (isrealvector(fhist) and fopt == np.min(fhist)):
        raise ValueError("fopt is not the minimum of fhist.")

    nhist = len(fhist)

    if "xhist" in output:
        xhist = output["xhist"]
        # Verify whether xhist is a real matrix of size.
        if not (isrealmatrix(xhist) and np.shape(xhist) == (len(xopt), nhist)):
            raise ValueError("output.xhist is not a real matrix.")

        # Check whether length(fhist) is equal to length(xhist) and nf respectively.
        if not (len(fhist) == xhist.shape[1] and xhist.shape[1] == nf):
            raise ValueError("length of fhist is not equal to length of xhist or nf.")

        # Check whether fhist == fun(xhist).
        fhist_eval = np.full((1, len(fhist)), np.nan)
        for i in range(len(fhist)):
            if xopt.ndim == 1:
                fhist_eval[i] = eval_fun(fun, xhist[:, i].T)
            else:
                fhist_eval[i] = eval_fun(fun, xhist[:, i])
