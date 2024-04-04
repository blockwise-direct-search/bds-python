import __init__


def isnumvec(x):
    is_numvec = (isinstance(x, list) or isinstance(x, __init__.np.ndarray)) and (
                __init__.np.ndim(x) == 1 or (__init__.np.ndim(x) == 2 and (__init__.np.shape(x)[0] == 1 or __init__.np.shape(x)[1] == 1)))
    return is_numvec
