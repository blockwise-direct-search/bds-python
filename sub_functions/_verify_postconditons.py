import numpy as np
from ._eval_fun import eval_fun
from ._isrealvector import isrealvector
from ._isrealmatrix import isrealmatrix
import pdb

def verify_postconditions(fun, xopt, fopt, exitflag, output):
    """
    Verify postconditions for optimization results.
    """
    
    # Verify whether xopt is a real vector.
    if not isrealvector(xopt):
        raise ValueError("xopt is not a real vector.")
    
    # Verify whether fopt is a real number.
    if not isinstance(fopt, float):
        raise ValueError("fopt is not a real number.")

    # Verify whether fun(xopt) == fopt
    if not np.isclose(eval_fun(fun, xopt)[0], fopt):  # Using np.isclose for floating-point comparison
        raise ValueError("fun(xopt) is not equal to fopt.")
    
    # Verify whether exitflag is an integer.
    if not isinstance(exitflag, int):
        raise ValueError("exitflag is not an integer.")
    
    # Verify whether nf is a positive integer.
    if 'funcCount' not in output:
        raise ValueError("output.funcCount does not exist.")
    nf = output['funcCount']

    # Check if nf is a positive integer
    if not isinstance(nf, int) or nf <= 0:
        raise ValueError("output.funcCount is not a positive integer.")
    
    # Verify whether output is a dictionary (structure).
    if not isinstance(output, dict):
        raise ValueError("output is not a structure.")
    
    # Verify whether output.fhist exists.
    if 'fhist' not in output:
        raise ValueError("output.fhist does not exist.")
    fhist = output['fhist']
    
    # Verify whether fopt is the minimum of fhist.
    if not (isrealvector(fhist) and fopt == np.min(fhist)):
        pass
        # raise ValueError("fopt is not the minimum of fhist.")
    
    nhist = len(fhist)

    if 'xhist' in output:
        xhist = output['xhist']
        # Verify whether xhist is a real matrix of correct size.
        if not (isrealmatrix(xhist) and (xhist.shape[0] == len(xopt))):
            raise ValueError("output.xhist is not a real matrix.")
        
        # Check whether length(fhist) is equal to length(xhist) and nf respectively.
        if not (len(fhist) == xhist.shape[1] == nf):
            raise ValueError("length of fhist is not equal to length of xhist or nf.")
        
        # Check whether fhist == fun(xhist).
        fhist_eval = np.full(len(fhist), np.nan)
        for i in range(len(fhist)):
            if xopt.ndim == 1:  # xopt is a row vector
                fhist_eval[i] = eval_fun(fun, xhist[:, i])[0]
            else:
                fhist_eval[i] = eval_fun(fun, xhist[:, i])[0]
        
        # In case of fhist_eval(i) = NaN or fhist(i) = NaN.
        assert np.all((np.isnan(fhist) & np.isnan(fhist_eval)) | (fhist == fhist_eval))