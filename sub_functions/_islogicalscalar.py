def islogicalscalar(value):
    r"""
    ISLOGICALSCALAR checks whether the input is a logical scalar.
    N.B.: islogicalscalar([]) = FALSE !!!
    """
    return isinstance(value, bool)