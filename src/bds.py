import numpy as np









def bds(fun, x0, options=None):

    r"""
    Minimize unconstrained optimization problems using blockwise direct search
    methods.

    Parameters
    ----------
    fun : {callable, None}
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple. If `fun`
        is ``None``, the objective function is assumed to be the zero function,
        resulting in a feasibility problem.
    x0 : array_like, shape (n,)
        Initial guess.

    options : dict, optional
        Options passed to the solver. Accepted keys are:

            debug_flag : bool, optional
                If debug_flag is true, the process will run in debug mode. 
                Otherwise, the process will run without debugging.
            Algorithm : str, optional
                Algorithm to use. It can be ``"cbds"`` (cyclic blockwise direct
                search), ``"pbds"`` (randomly permuted blockwise direct search),
                ``"rbds"`` (randomized blockwise direct search), ``"ds"`` (the
                classical direct search), ``"pads"`` (parallel blockwise direct
                search), or ``"scbds"`` (symmetric blockwise direct search).
                Default is ``"cbds"``.
            num_blocks : int, optional
                Number of blocks. A positive integer. Default is ``n`` if
                ``Algorithm`` is ``"cbds"``, ``"pbds"``, or ``"rbds"``, and 1 if
                ``Algorithm`` is ``"ds"``.
            MaxFunctionEvaluations : int, optional
                Maximum number of function evaluations. Default is ``500 * n``.
            direction_set : array_like, shape (n, n), optional
                A matrix whose columns will be used to define the polling
                directions. If ``options`` does not contain ``direction_set``,
                then the polling directions will be {e_1, -e_1, ..., e_n, -e_n}.
                Otherwise, it should be a nonsingular n-by-n matrix. Then the
                polling directions will be {d_1, -d_1, ..., d_n, -d_n}, where
                d_i is the i-th column of ``direction_set``. If ``direction_set``
                is not singular, then we will revise the ``direction_set`` to
                make it linear independent. Default is the identity matrix.
                See `get_direction_set` for details.
            expand : float, optional
                Expanding factor of step size. A real number no less than 1.
                Default is ``2.0``.
            shrink : float, optional
                Shrinking factor of step size. A positive number less than 1.
                Default is ``0.5``.
            forcing_function : callable, optional
                The forcing function used for deciding whether the step achieves
                a sufficient decrease. Default is ``lambda alpha: alpha**2``.
                See also ``reduction_factor``.
            reduction_factor : array_like, shape (3,), optional
                Factors multiplied to the forcing function when deciding whether
                the step achieves a sufficient decrease. A 3-dimensional vector
                such that ``reduction_factor[0] <= reduction_factor[1] <=
                reduction_factor[2]``, ``reduction_factor[0] >= 0``, and
                ``reduction_factor[1] > 0``. ``reduction_factor[0]`` is used for
                deciding whether to update the base point, 
                ``reduction_factor[1]`` is used for deciding whether to shrink
                the step size, and ``reduction_factor[2]`` is used for deciding
                whether to expand the step size. Default is 
                ``[0.0, numpy.finfo(float).eps, numpy.finfo(float).eps]``. See
                also ``forcing_function``.
            StepTolerance : float, optional
                Lower bound of the step size. If the step size is smaller than
                ``StepTolerance``, then the algorithm terminates. A (small)
                positive number. Default is ``1e-10``.
            ftarget : float, optional
                Target of the function value. If the function value is smaller
                than or equal to this target, then the algorithm terminates.
                Default is ``-numpy.inf``.
            polling_inner : str, optional
                Polling strategy in each block. It can be ``"complete"`` or
                ``"opportunistic"``. Default is ``"opportunistic"``.
            cycling_inner : int, optional
                Cycling strategy employed within each block. It is used only
                when ``polling_inner`` is ``"opportunistic"``. It can be 0, 1,
                2, 3, 4. See `cycling` for details. Default is ``3``.
            with_cycling_memory : bool, optional
                Whether the cycling strategy within each block memorizes the
                history or not. It is used only when ``polling_inner`` is
                ``"opportunistic"``. Default is ``True``.
            permuting_period : int, optional
                It is only used in PBDS, which shuffles the blocks every
                ``permuting_period`` iterations. A positive integer. Default is
                ``1``.
            replacement_delay : int, optional
                It is only used for RBDS. Suppose that ``replacement_delay`` is
                ``r``. If block ``i`` is selected at iteration ``k``, then it
                will not be selected at iterations ``k+1``, ..., ``k+r``. An
                integer between 0 and ``num_blocks-1``. Default is ``0``.
            seed : int, optional
                The seed for permuting blocks in PBDS or randomly choosing one
                block in RBDS. It is only for reproducibility in experiments. A
                positive integer.
            output_xhist : bool, optional
                Whether to output the history of points visited. Default is
                ``False``.
            output_alpha_hist : bool, optional
                Whether to output the history of step sizes. Default is
                ``False``.
            output_block_hist : bool, optional
                Whether to output the history of blocks visited. Default is
                ``False``.
            verbose : bool, optional
                a flag deciding whether to print during the computation.
                Default is ``False``, which means no printing. If ``verbose``
                is True, then the function values, the corresponding point,
                and the step size will be printed in each function evaluation.

    Returns
    -------
    `scipy.optimize.OptimizeResult`
        Result of the optimization procedure, with the following fields:

            message : str
                The information of the exitflag.
            funcCount : int
                The number of function evaluations.
            xopt : `numpy.ndarray`, shape (n,)
                Solution point.
            fopt : float
                Objective function value at the solution point.


        If ``output_xhist`` is True, the result also has the following fields:

            xhist : `numpy.ndarray`, history of points visited, 
                    shape (n, MaxFunctionEvaluations)
        If ``output_alpha_hist`` is True, the result also has the following 
        fields:
            alpha_hist : `numpy.ndarray`, history of step sizes for every
                            iteration
        If ``output_block_hist`` is True, the result also has the following
        fields:
            block_hist : `numpy.ndarray`, history of blocks visited

        A description of the termination statuses is given below.

        .. list-table::
            :widths: 25 75
            :header-rows: 1

            * - Exit status
              - Description
            * - 0
              - The StepTolearance of the step size has been reached.
            * - 1
              - The target objective function value has been reached.
            * - 2
              - The maximum number of function evaluations is reached.
            * - 3
              - The maximum number of iterations is reached.

    References
    ----------
    .. [1] Li, H. and Zhang, Z. *Blockwise direct search methods*. GitHub repo.
       Department of Applied Mathematics, The Hong Kong Polytechnic University,
       Hong Kong, China, 2023.        
       URL: https://github.com/blockwise-direct-search/bds.
    """

    # If FUN is a string, then convert it to a function handle.
    if __init__.ischarstr(fun):
        fun = eval(fun)
    # Redefine fun to accept column vectors if x0 is a row vector, as we use column vectors internally.
    fun_orig = fun

    # If x0 is a list, then convert it to a numpy vector.
    if isinstance(x0, list):
        # Convert list to numpy array.
        x0_array = __init__.np.array(x0)
        # Convert numpy array to numpy vector.
        x0 = x0_array.reshape((-1, 1))
    elif isinstance(x0, __init__.np.ndarray):
        if not __init__.isrealvector(x0)[0]:
            raise ValueError("If x0 is a numpy array, x0 must be a real vector.")

    # Transpose x0 if it is a row vector.
    x0_is_row = __init__.is_row(x0)
    if x0_is_row:
        x0 = x0.reshape(-1, 1)

        def fun(x):
            return fun_orig(x.T)

    # Get basic options that are needed for the initialization.
    if options is None:
        options = {}
    else:
        options = dict(options)

    # Set the default value of debug_flag. If options do not contain debug_flag, then
    # debug_flag is set to false.
    if "debug_flag" in options:
        debug_flag = options["debug_flag"]
    else:
        debug_flag = False

    if debug_flag:
        print("x0_is_row: ", x0_is_row)
        __init__.verify_preconditions(fun, x0, options)

    if not "seed" in options:
        options["seed"] = list(range(624))
    random_stream = __init__.np.random.default_rng()
    random_stream.shuffle(options["seed"])

    n = len(x0)
    if not "Algorithm" in options:
        options["Algorithm"] = __init__.get_default_constant("Algorithm")

    if "direction_set_type" in options:
        if options["direction_set_type"].lower() == "randomized_orthogonal":
            random_matrix = __init__.np.random.randn(n * n).reshape(n, n)
            options["direction_set"], _ = __init__.np.linalg.qr(random_matrix)
        elif options["direction_set_type"].lower() == "randomized":
            options["direction_set"] = __init__.np.random.randn(n, n)

    D = __init__.get_direction_set(n, options)

    num_directions = D.shape[1]
    if options["Algorithm"].lower() == "ds":
        num_blocks = 1
    elif "block" in options:
        num_blocks = min(num_directions, options["block"])
    elif options["Algorithm"].lower() in ["cbds", "pbds", "rbds", "pads", "scbds"]:
        num_blocks = __init__.math.ceil(num_directions / 2)
    else:
        raise ValueError("The field Algorithm of options is invalid.")

    direction_set_indices = __init__.divide_direction_set(n, num_blocks, debug_flag)

    if "expand" in options:
        expand = options["expand"]
    else:
        expand = __init__.get_default_constant("expand")

    if "shrink" in options:
        shrink = options["shrink"]
    else:
        shrink = __init__.get_default_constant("shrink")

    if "reduction_factor" in options:
        reduction_factor = options["reduction_factor"]
    else:
        reduction_factor = __init__.get_default_constant("reduction_factor")

    if "forcing_function" in options:
        forcing_function = options["forcing_function"]
    else:
        forcing_function = __init__.get_default_constant("forcing_function")

    if "forcing_function_type" in options:
        if options["forcing_function_type"] == "quadratic":

            def forcing_function(x):
                return x ** 2
        elif options["forcing_function_type"] == "cubic":

            def forcing_function(x):
                return x ** 3

    if "polling_inner" in options:
        polling_inner = options["polling_inner"]
    else:
        polling_inner = __init__.get_default_constant("polling_inner")

    if "cycling_inner" in options:
        cycling_inner = options["cycling_inner"]
    else:
        cycling_inner = __init__.get_default_constant("cycling_inner")

    if options["Algorithm"].lower() == "pbds":
        if "permuting_period" in options:
            permuting_period = options["permuting_period"]

    if options["Algorithm"].lower() == "rbds":
        if "replacement_delay" in options:
            replacement_delay = min(options["replacement_delay"], num_blocks - 1)
        else:
            replacement_delay = min(__init__.get_default_constant("replacement_delay"), num_blocks - 1)

    if "with_cycling_memory" in options:
        with_cycling_memory = options["with_cycling_memory"]
    else:
        with_cycling_memory = __init__.get_default_constant("with_cycling_memory")

    if "MaxFunctionEvaluations" in options:
        MaxFunctionEvaluations = options["MaxFunctionEvaluations"]
    else:
        MaxFunctionEvaluations = __init__.get_default_constant("MaxFunctionEvaluations_dim_factor") * n

    maxit = MaxFunctionEvaluations

    if "StepTolerance" in options:
        alpha_tol = options["StepTolerance"]
    else:
        alpha_tol = __init__.get_default_constant("StepTolerance")

    if "ftarget" in options:
        ftarget = options["ftarget"]
    else:
        ftarget = __init__.get_default_constant("ftarget")

    if "output_alpha_hist" in options:
        output_alpha_hist = options["output_alpha_hist"]
    else:
        output_alpha_hist = __init__.get_default_constant("output_alpha_hist")
    if output_alpha_hist:
        try:
            alpha_hist = __init__.np.full((num_blocks, maxit), __init__.np.nan)
        except Exception as e:
            print("An exception occurred:", str(e))
            output_alpha_hist = False

    if "alpha_init" in options:
        if len(options["alpha_init"]) == 1:
            alpha_all = options["alpha_init"] * __init__.np.ones((num_blocks, 1))
        elif len(options["alpha_init"]) == num_blocks:
            alpha_all = options["alpha_init"]
        else:
            raise ValueError("The length of alpha_init should be equal to num_blocks or equal to 1.")
    elif num_blocks == n and D.shape[1] == 2 * n and "alpha_init_scaling" in options and options["alpha_init_scaling"]:
        x0_coordinates = __init__.np.linalg.lstsq(D[:, 0:2:2 * n - 1], x0, rcond=None)[0]
        x0_scales = __init__.np.abs(x0_coordinates)
        if "alpha_init_scaling_factor" in options:
            alpha_all = options["alpha_init_scaling_factor"] * x0_scales
        else:
            alpha_all = 0.5 * __init__.np.maximum(1, __init__.np.abs(x0_scales))
    else:
        alpha_all = __init__.np.ones((num_blocks, 1))

    fopt_all = __init__.np.full((1, num_blocks), __init__.np.nan)
    xopt_all = __init__.np.full((n, num_blocks), __init__.np.nan)

    fhist = __init__.np.full((1, MaxFunctionEvaluations), __init__.np.nan)

    if "output_xhist" in options:
        output_xhist = options["output_xhist"]
    else:
        output_xhist = __init__.get_default_constant("output_xhist")
    if output_xhist:
        try:
            xhist = __init__.np.full((n, maxit), __init__.np.nan)
        except Exception as e:
            print("An exception occurred:", str(e))
            output_xhist = False

    if "output_block_hist" in options:
        output_block_hist = options["output_block_hist"]
    else:
        output_block_hist = __init__.get_default_constant("output_block_hist")
    block_hist = __init__.np.full((1, maxit), __init__.np.nan)

    if "verbose" in options:
        verbose = options["verbose"]
    else:
        verbose = __init__.get_default_constant("verbose")

    exitflag = __init__.get_exitflag("MAXIT_REACHED", debug_flag)

    xbase = x0
    [fbase, fbase_real] = __init__.eval_fun(fun, xbase)
    __init__.pdb.set_trace()
    if verbose:
        print("Function number %d, F = %f" % (1, fbase))
        print("The corresponding X is:")
        print(" ".join(map(str, xbase.flatten())))
        print()

    xopt = xbase
    fopt = fbase

    nf = 1
    if output_xhist:
        xhist = __init__.np.full((n, maxit), __init__.np.nan)
        xhist[:, nf - 1] = xbase

    fhist[nf - 1] = fbase_real

    terminate = False
    if fopt <= ftarget:
        information = "FTARGET_REACHED"
        exitflag = __init__.get_exitflag(information, debug_flag)
        maxit = 0

    if "block_indices_permuted_init" in options and options["block_indices_permuted_init"]:
        all_block_indices = __init__.np.random.permutation(num_blocks)
    else:
        all_block_indices = __init__.np.arange(1, num_blocks + 1)
    num_visited_blocks = 0

    for iteration in range(1, maxit + 1):

        if options["Algorithm"].lower() in ["ds", "cbds", "pads"]:
            block_indices = all_block_indices
        elif options["Algorithm"].lower() == "pbds" and (iteration - 1) % permuting_period == 0:
            block_indices = __init__.np.random.permutation(num_blocks)
        elif options["Algorithm"].lower() == "rbds":
            unavailable_block_indices = block_hist[max(1, iteration - replacement_delay): iteration - 1]
            available_block_indices = __init__.np.setdiff1d(all_block_indices, unavailable_block_indices)
            idx = __init__.np.random.randint(len(available_block_indices))
            block_indices = available_block_indices[idx]  # a vector of length 1
        elif options["Algorithm"].lower() == "scbds":
            block_indices = __init__.np.concatenate([all_block_indices, __init__.np.arange(num_blocks - 1, 1, -1)])

        for i in range(1, len(block_indices) + 1):
            i_real = block_indices[i - 1] - 1

            direction_indices = direction_set_indices[i_real]

            suboptions = {"FunctionEvaluations_exhausted": nf, "MaxFunctionEvaluations": MaxFunctionEvaluations - nf,
                          "cycling_inner": cycling_inner, "with_cycling_memory": with_cycling_memory,
                          "reduction_factor": reduction_factor, "forcing_function": forcing_function,
                          "ftarget": ftarget, "polling_inner": polling_inner, "verbose": verbose,
                          "debug_flag": debug_flag}

            [sub_xopt, sub_fopt, sub_exitflag, sub_output] = __init__.inner_direct_search(fun, xbase,
                                                                                          fbase,
                                                                                          D[:, direction_indices],
                                                                                          direction_indices,
                                                                                          alpha_all[i_real], suboptions)

            num_visited_blocks = num_visited_blocks + 1
            block_hist[num_visited_blocks] = i_real

            if output_alpha_hist:
                alpha_hist[:, iteration] = alpha_all

            if output_xhist:
                xhist[:, (nf + 1): (nf + sub_output.nf)] = sub_output.xhist

            fhist[(nf + 1): (nf + sub_output.nf)] = sub_output.fhist
            nf = nf + sub_output["nf"]

            if sub_fopt + reduction_factor[2] * forcing_function(alpha_all[i_real]) < fbase:
                alpha_all[i_real] = expand * alpha_all[i_real]
            elif sub_fopt + reduction_factor[1] * forcing_function(alpha_all[i_real]) >= fbase:
                alpha_all[i_real] = shrink * alpha_all[i_real]

            fopt_all[i_real] = sub_fopt
            xopt_all[:, i_real] = sub_xopt

            if options["Algorithm"].lower() != "pads":
                if (reduction_factor[0] <= 0 and sub_fopt < fbase) or sub_fopt + reduction_factor[0] * forcing_function(
                        alpha_all[i_real]) < fbase:
                    xbase = sub_xopt
                    fbase = sub_fopt

            direction_set_indices[i_real] = sub_output["direction_indices"]

            if sub_output["terminate"]:
                terminate = True
                exitflag = sub_exitflag
                break

            if max(alpha_all) < alpha_tol:
                terminate = True
                exitflag = __init__.get_exitflag("SMALL_ALPHA", debug_flag)
                break

        _, index = __init__.np.nanmin(fopt_all)
        if fopt_all[index] < fopt:
            fopt = fopt_all[index]
            xopt = xopt_all[:, index]

        if options["Algorithm"].lower() == "pads":
            if (reduction_factor[0] <= 0 and fopt < fbase) or fopt + reduction_factor[0] * forcing_function(
                    min(alpha_all)) < fbase:
                xbase = xopt
                fbase = fopt

        if terminate:
            break

    output = {}
    output["funcCount"] = nf

    if output_block_hist:
        output["blocks_hist"] = block_hist[:num_visited_blocks]
    if output_alpha_hist:
        output["alpha_hist"] = alpha_hist[:, :min(iteration, maxit)]
    if output_xhist:
        output["xhist"] = xhist[:, :nf]
    output["fhist"] = fhist[:nf]

    exitflag_value = exitflag
    if exitflag_value == __init__.get_exitflag("SMALL_ALPHA", debug_flag):
        output["message"] = "The StepTolerance of the step size is reached."
    elif exitflag_value == __init__.get_exitflag("MAXFUN_REACHED", debug_flag):
        output["message"] = "The maximum number of function evaluations is reached."
    elif exitflag_value == __init__.get_exitflag("FTARGET_REACHED", debug_flag):
        output["message"] = "The target of the objective function is reached."
    elif exitflag_value == __init__.get_exitflag("MAXIT_REACHED", debug_flag):
        output["message"] = "The maximum number of iterations is reached."
    else:
        output["message"] = "Unknown exitflag"

    if x0_is_row:
        xopt = xopt.T

    if debug_flag:
        __init__.verify_postconditions(fun_orig, xopt, fopt, exitflag, output)

    return xopt, fopt, exitflag, output


def _eval_fun(fun, x):
    # EVAL_FUN evaluates function FUN at point X. If FUN is not well defined at X, return NaN.
    try:
        f_real = fun(x)
    except:
        print('The function evaluation failed.')
        f_real = np.nan

    # Apply the moderate extreme barrier to handle NaN, huge values, and evaluation failures.
    if __init__.np.isnan(f_real):
        f_real = __init__.np.inf
    f = min(f_real, 2 ** 100, __init__.np.sqrt(__init__.np.finfo(float).max))

    return f, f_real
