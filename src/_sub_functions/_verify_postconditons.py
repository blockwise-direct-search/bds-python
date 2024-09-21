import __init__


def verify_postconditions(fun, xopt, fopt, exitflag, output):
    # Verify whether xopt is a real vector.
    if not __init__.isrealvector(xopt):
        raise ValueError("xopt is not a real vector.")

    # Verify whether fopt is a real number.
    if not __init__.isrealscalar(fopt):
        raise ValueError("fopt is not a real number.")

    # Verify whether fun(xopt) == fopt.
    # For CUTest problem, function handle only accept column vector.
    if not __init__.eval_fun(fun, xopt) == fopt:
        raise ValueError("fun(xopt) is not equal to fopt.")

    # Verify whether exitflag is an integer.
    if not __init__.isintegerscalar(exitflag):
        raise ValueError("exitflag is not an integer.")

    # Verify whether nf is a positive integer.
    if "funcCount" not in output:
        raise ValueError("funcCount does not exist in output.")
    nf = output["funcCount"]
    if not (__init__.isintegerscalar(nf) and nf > 0):
        raise ValueError("funcCount is not a positive integer.")

    # Verify whether output is a dictionary.
    if not isinstance(output, dict):
        raise ValueError("output is not a dictionary.")

    # Verify whether output.fhist exists.
    if "fhist" not in output:
        raise ValueError("output.fhist does not exist.")
    fhist = output["fhist"]
    # Verify whether fopt is the minimum of fhist.
    if not (__init__.isrealvector(fhist) and fopt == __init__.np.min(fhist)):
        raise ValueError("fopt is not the minimum of fhist.")

    nhist = len(fhist)

    if "xhist" in output:
        xhist = output["xhist"]
        # Verify whether xhist is a real matrix of size.
        if not (__init__.isrealmatrix(xhist) and __init__.np.shape(xhist) == (len(xopt), nhist)):
            raise ValueError("output.xhist is not a real matrix.")

        # Check whether length(fhist) is equal to length(xhist) and nf respectively.
        if not (len(fhist) == xhist.shape[1] and xhist.shape[1] == nf):
            raise ValueError("length of fhist is not equal to length of xhist or nf.")

        # Check whether fhist == fun(xhist).
        fhist_eval = __init__.np.full((1, len(fhist)), __init__.np.nan)
        for i in range(len(fhist)):
            if xopt.ndim == 1:
                fhist_eval[i] = __init__.eval_fun(fun, xhist[:, i].T)
            else:
                fhist_eval[i] = __init__.eval_fun(fun, xhist[:, i])
