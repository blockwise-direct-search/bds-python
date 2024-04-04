import __init__


# ISREALCOLUMN checks whether x is a real column. If yes, it returns len = length(x); otherwise, len = NaN.
def isrealcolumn(x):
    if __init__.np.size(x) == 0:
        isrc = True
        length = 0
    elif (x.size == 1 and x.ndim == 1) or (__init__.np.ndim(x) == 2 and x.shape[1] == 1):
        isrc = True
        length = len(x)
    else:
        isrc = False
        length = __init__.np.nan

    return isrc, length
