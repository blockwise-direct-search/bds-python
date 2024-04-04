import __init__

# ISLOGICALSCALAR checks whether x is a logical scalar, including 0 and 1.
def islogicalscalar(x):

    if isinstance(x, bool) and __init__.np.isscalar(x):
        isis = True
    elif __init__.isrealscalar(x) and (x == 1 or x == 0):
        isis = True
    else:
        isis = False

        return isis
