import __init__


# ISREALSCALAR checks whether x is a real scalar.
def isrealscalar(x):
    isrs = __init__.np.isscalar(x) and __init__.np.isreal(x) and __init__.np.ndim(x) == 0
    return isrs
