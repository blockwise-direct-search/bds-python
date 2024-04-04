import __init__


def is_row(x):
    x_is_row = False
    if isinstance(x, __init__.np.ndarray) and x.ndim == 1:
        x_is_row = True
    return x_is_row
