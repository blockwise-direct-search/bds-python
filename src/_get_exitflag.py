import __init__


def get_exitflag(information, debug_flag):
    # GET_EXITFLAG gets the EXITFLAG of BDS.
    #   SMALL_ALPHA     Step size is below StepTolerance. In the case of variable step sizes,
    #                   SMALL_ALPHA indicates the largest component of step sizes is
    #                   below StepTolerance.
    #   MAXFUN_REACHED  The number of function evaluations reaches MAXFUN.
    #   FTARGET_REACHED Function value is smaller than or equal to FTARGET.
    #   MAXIT_REACHED   The number of iterations reaches MAXIT.

    # Check whether INFORMATION is a string or not.
    if debug_flag:
        if not isinstance(information, str):
            raise ValueError("Information is not a string.")

    if information == "SMALL_ALPHA":
        exitflag = 0
    elif information == "FTARGET_REACHED":
        exitflag = 1
    elif information == "MAXFUN_REACHED":
        exitflag = 2
    elif information == "MAXIT_REACHED":
        exitflag = 3
    else:
        exitflag = -1

    # exitflag = find(break_conditions == information) - 1;
    if not exitflag:
        exitflag = -1
        print("New break condition happens.")

    # Check whether EXITFLAG is an integer or not.
    if debug_flag:
        if not __init__.isintegerscalar(exitflag):
            raise ValueError("Exitflag is not an integer.")

    return exitflag
