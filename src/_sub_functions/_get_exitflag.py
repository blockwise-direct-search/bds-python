def get_exitflag(information):
    """
    Get the EXITFLAG of BDS.

    Parameters:
        information (str): The information string indicating the exit condition.

    Returns:
        int: The exit flag corresponding to the information.
            0: SMALL_ALPHA - Step size is below StepTolerance.
            1: FTARGET_REACHED - Function value is smaller than or equal to FTARGET.
            2: MAXFUN_REACHED - The number of function evaluations reaches MAXFUN.
            3: MAXIT_REACHED - The number of iterations reaches MAXIT.
            -1: Unknown condition.
    """

    # Check whether INFORMATION is a string
    if not isinstance(information, str):
        raise ValueError("Information is not a string.")

    # Determine exit flag based on the information provided
    if information == "SMALL_ALPHA":
        exitflag = 0
    elif information == "FTARGET_REACHED":
        exitflag = 1
    elif information == "MAXFUN_REACHED":
        exitflag = 2
    elif information == "MAXIT_REACHED":
        exitflag = 3
    else:
        exitflag = -1  # Unknown condition

    # Check if exitflag is -1 and display a message
    if exitflag == -1:
        print("New break condition happens.")

    # Check if EXITFLAG is an integer
    if not isinstance(exitflag, int):
        raise ValueError("Exitflag is not an integer.")

    return exitflag