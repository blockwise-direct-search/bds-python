import __init__


def isrealmatrix(x):
    if __init__.np.size(x) == 0:
        isrc = True
        m = 0
        n = 0
    elif __init__.np.issubdtype(x.dtype, __init__.np.number) and __init__.np.isreal(x) and x.ndim <= 2:
        isrc = True
        m, n = x.shape
    else:
        isrc = False
        m = __init__.np.nan
        n = __init__.np.nan
