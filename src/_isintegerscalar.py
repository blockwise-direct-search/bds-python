import __init__


# ISINTEGERSCALAR checks whether x is an integer scalar.
def isintegerscalar(x):
    isis = __init__.isrealscalar(x) and (x % 1 == 0)
    return isis
