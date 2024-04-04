import __init__


def eval_fun(fun, x):
    # EVAL_FUN evaluates function FUN at point X. If FUN is not well defined at X, return NaN.
    try:
        f_real = fun(x)
    except:
        print('The function evaluation failed.')
        f_real = __init__.np.nan

    # Apply the moderate extreme barrier to handle NaN, huge values, and evaluation failures.
    if __init__.np.isnan(f_real):
        f_real = __init__.np.inf
    f = min(f_real, 2 ** 100, __init__.np.sqrt(__init__.np.finfo(float).max))

    return f, f_real
