def isintegerscalar(value):
   
    r"""
    ISINTEGERSCALAR checks whether x is an integer scalar.
    N.B.: isintegerscalar([]) = FALSE, isintegerscalar(NaN) = FALSE, 
    isintegerscalar(inf) = FALSE !!!
    """
    return isinstance(value, int) and not isinstance(value, bool)
