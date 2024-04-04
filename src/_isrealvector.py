import __init__


def isrealvector(x):
    if __init__.isrealrow(x)[0] or __init__.isrealcolumn(x)[0]:
        isrv = True
        length = len(x)
    else:
        isrv = False
        length = __init__.np.nan
    return isrv, length
