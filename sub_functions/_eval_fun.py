import numpy as np
import pdb

def eval_fun(fun, x):
    r'''
    EVAL_FUN evaluates function FUN at point X, returning f and f_real.
    f_real is the real function value, while f is a moderated version of f_real.
    The moderation is to handle NaN, huge values, and evaluation failures. The 
    algorithm will operate on f, while f_real is used for recording the history.
    '''
    try:
        f_real = fun(x)
    except:
        print('The function evaluation failed.')
        f_real = np.nan

    # Apply the moderate extreme barrier to handle NaN, huge values, and 
    # evaluation failures.
    # See 4.5 of "PDFO: A Cross-Platform Package for Powell's Derivative-Free 
    # Optimization Solvers" by Tom M. Ragonneau and Zaikun Zhang.
    if np.isnan(f_real):
        f_real = np.inf
    f = min(f_real, 2 ** 100, np.sqrt(np.finfo(float).max))

    return f, f_real
