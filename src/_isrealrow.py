import __init__

def isrealrow(x):
    if not isinstance(x, __init__.np.ndarray):
        print("arr is a NumPy array")
    else:
        print("arr is not a NumPy array")


    if __init__.np.size(x) == 0:
        isrr = True
        length = 0
    elif (x.size == 1 and x.ndim == 1) or __init__.np.ndim(x) == 1:
        isrr = True
        length = len(x)
    else:
        isrr = False
        length = __init__.np.nan
    return isrr, length
