import numpy as np
import pdb
from _sub_functions import *

def chrosen(x):
    """
    The objective function defined by the Rosenbrock function.

    Args:
        x (numpy.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    f = np.sum((x[:-1] - 1) ** 2 + 4 * (x[1:] - x[:-1] ** 2) ** 2)
    # Uncomment the line below if you want to add noise to the function value
    # f *= (1 + 1e-8 * np.random.randn())
    return f

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
    
    # Set options to an empty dictionary if it is not provided.
    if options is None:
        options = {}

    # Transpose x0 if it is a row.
    x0_is_row = isrow(x0)
    # Ensure x0 is a column vector. Inside the algorithm, we will treat the 
    # point as a numpy array.
    x0 = np.array(x0).reshape(-1, 1)

    # Get the dimension of the problem.
    n = len(x0)

    # Set the default value of debug_flag.
    if "debug_flag" not in options:
        debug_flag = False
    else:
        debug_flag = options["debug_flag"]

    # Get the direction set.
    D = get_direction_set(n, options)
    # Set the default Algorithm of BDS, which is "cbds".
    if "Algorithm" not in options:
        options["Algorithm"] = get_default_constant("Algorithm")

    # Get the number of blocks.
    num_directions = D.shape[1]
    if options["Algorithm"].lower() == "ds":
        num_blocks = 1
    elif "block" in options:
        num_blocks = min(num_directions, options["block"])
    elif options["Algorithm"].lower() in ["cbds", "pbds", "rbds", "pads", "scbds"]:
        num_blocks = int(np.ceil(num_directions / 2))
    
    # Determine the indices of directions in each block.
    direction_set_indices = divide_direction_set(n, num_blocks, debug_flag)

    options["num_blocks"] = num_blocks
    # Check the inputs of the user when debug_flag is true.
    if debug_flag:
        verify_preconditions(fun, x0, options)
    # If FUN is a string, then convert it to a function handle.
    if isinstance(fun, str):
    # Convert string to function handle.
        fun = globals()[fun]  
    # Redefine fun to accept columns if x0 is a row.
    fun_orig = fun
    pdb.set_trace()
    # TODO: why we use fun_orig here?
    if x0_is_row:
        fun = lambda x: fun_orig(x.ravel())
    pdb.set_trace()

    # Avoid randomized strings in options.
    if "seed" in options:
        seed = options["seed"]
    random_stream = np.random.default_rng(seed=None)  # 使用系统时间作为种子，类似于"shuffle"

    # Set the factors for expanding and shrinking the step sizes.
    if "expand" not in options:
        expand = get_default_constant("expand")
    else:
        expand = options["expand"]
    if "shrink" not in options:
        shrink = get_default_constant("shrink")
    else:
        shrink = options["shrink"]

    # Set the value of reduction_factor.
    if "reduction_factor" not in options:
        reduction_factor = get_default_constant("reduction_factor")
    else:
        reduction_factor = options["reduction_factor"]

    # Set the forcing function, which should be the function handle.
    if "forcing_function" not in options:
        forcing_function = get_default_constant("forcing_function")
    else:
        forcing_function = options["forcing_function"]

    # Set polling_inner and cycling_inner strategies.
    if "polling_inner" not in options:
        polling_inner = get_default_constant("polling_inner")
    else:
        polling_inner = options["polling_inner"]
    if "cycling_inner" not in options:
        cycling_inner = get_default_constant("cycling_inner")
    else:
        cycling_inner = options["cycling_inner"]

    # Set permuting_period for "pbds" algorithm.
    if options["Algorithm"].lower() == "pbds":
        if "permuting_period" not in options:
            permuting_period = get_default_constant("permuting_period")
        else:
            permuting_period = options["permuting_period"]

    # Set replacement_delay for "rbds" algorithm.
    if options["Algorithm"].lower() == "rbds":
        if "replacement_delay" not in options:
            replacement_delay = get_default_constant("replacement_delay")
        else:
            replacement_delay = options["replacement_delay"]

    # Set the boolean value of with_cycling_memory.
    if "with_cycling_memory" not in options:
        with_cycling_memory = get_default_constant("with_cycling_memory")
    else:
        with_cycling_memory = options["with_cycling_memory"]

    # Set the maximum number of function evaluations.
    if "MaxFunctionEvaluations" not in options:
        MaxFunctionEvaluations = get_default_constant("MaxFunctionEvaluations_dim_factor") * n
    else:
        MaxFunctionEvaluations = options["MaxFunctionEvaluations"]

    # Set the maximum number of iterations.
    maxit = MaxFunctionEvaluations

    # Set the value of StepTolerance.
    if "StepTolerance" not in options:
        alpha_tol = get_default_constant("StepTolerance")
    else:
        alpha_tol = options["StepTolerance"]

    # Set the target of the objective function.
    if "ftarget" not in options:
        ftarget = get_default_constant("ftarget")
    else:
        ftarget = options["ftarget"]

    # Decide whether to output the history of step sizes.
    if "output_alpha_hist" not in options:
        output_alpha_hist = get_default_constant("output_alpha_hist")
    else:
        output_alpha_hist = options["output_alpha_hist"]

    if output_alpha_hist:
        try:
            alpha_hist = np.full((num_blocks, maxit), np.nan)
        except MemoryError:
            output_alpha_hist = False
            print("alpha_hist will not be included in the output due to the memory limit.")
    
    # Set the initial step sizes.
    if "alpha_init" in options:
        if len(options["alpha_init"]) == 1:
            alpha_all = np.full(num_blocks, options["alpha_init"])
        elif len(options["alpha_init"]) == num_blocks:
            alpha_all = np.array(options["alpha_init"])
        elif options["alpha_init"].lower() == "auto":
            # Calculate x0_coordinates based on D
            x0_coordinates = np.linalg.lstsq(D[:, 0:2 * num_blocks - 1:2], x0, rcond=None)[0]
            alpha_all = 0.5 * np.maximum(1, np.abs(x0_coordinates))
    else:
        alpha_all = np.ones(num_blocks)

    # Initialize best function values and corresponding x values for each block.
    fopt_all = np.full(num_blocks, np.nan)
    xopt_all = np.full((len(x0), num_blocks), np.nan)

    # Initialize the history of function values.
    fhist = np.full(MaxFunctionEvaluations, np.nan)

    # Initialize the history of points visited.
    if "output_xhist" not in options:
        output_xhist = get_default_constant("output_xhist")
    else:
        output_xhist = options["output_xhist"]
    if output_xhist:
        try:
            xhist = np.full((len(x0), MaxFunctionEvaluations), np.nan)
        except MemoryError:
            output_xhist = False
            print("xhist will not be included in the output due to memory limits.")

    # Decide whether to output the history of blocks visited.
    if "output_block_hist" not in options:
        output_block_hist = get_default_constant("output_block_hist")
    else:
        output_block_hist = options["output_block_hist"]

    # Decide whether to print during the computation.
    if "verbose" not in options:
        verbose = get_default_constant("verbose")
    else:
        verbose = options["verbose"]

    # Initialize the history of blocks visited.
    block_hist = np.full(MaxFunctionEvaluations, np.nan)

    # Initialize exitflag.
    exitflag = get_exitflag("MAXIT_REACHED")

    # Initialize base point and its function value.
    xbase = x0
    pdb.set_trace()
    fbase, fbase_real = eval_fun(fun, xbase)

    if verbose:
        print(f"Function number 1, F = {fbase_real:.6f}")
        print("The corresponding X is:")
        print(" ".join(f"{val:.6f}" for val in xbase))

    # Initialize best point and function value.
    xopt = xbase
    fopt = fbase

    # Initialize function evaluations count and history.
    nf = 1
    if output_xhist:
        xhist[:, nf - 1] = xbase

    fhist[nf - 1] = fbase_real

    terminate = False
    # Check whether FTARGET is reached.
    if fopt <= ftarget:
        information = "FTARGET_REACHED"
        exitflag = get_exitflag(information)

        # If FTARGET is reached at the first evaluation, no further iterations.
        maxit = 0

    # Initialize block indices.
    all_block_indices = np.arange(1, num_blocks + 1)
    num_visited_blocks = 0
    
    # Initialize the output structure
    output = {
        "funcCount": 0,
        "blocks_hist": None,
        "alpha_hist": None,
        "xhist": None,
        "fhist": None,
        "message": ""
    }
    
    for iter in range(1, maxit + 1):
        # Define block_indices based on the algorithm
        if options["Algorithm"].lower() in ["ds", "cbds", "pads"]:
            block_indices = all_block_indices
        elif options["Algorithm"].lower() == "pbds" and (iter - 1) % options["permuting_period"] == 0:
            block_indices = random_stream.permutation(len(all_block_indices)) + 1  # 1-based index
        elif options["Algorithm"].lower() == "rbds":
            unavailable_block_indices = block_hist[max(0, iter - replacement_delay):iter]
            available_block_indices = np.setdiff1d(all_block_indices, unavailable_block_indices)
            idx = random_stream.integers(len(available_block_indices))
            block_indices = available_block_indices[idx]
        elif options["Algorithm"].lower() == "scbds":
            block_indices = np.concatenate((all_block_indices, (num_blocks - 1) - np.arange(1, num_blocks - 1)[::-1]))

        for i in range(len(block_indices)):
            i_real = block_indices[i]  # Real index of the block to visit
            
            # Get indices of directions in the block
            direction_indices = direction_set_indices[i_real]
            
            # Set options for the direct search in the block
            suboptions = {
                "FunctionEvaluations_exhausted": nf,
                "MaxFunctionEvaluations": MaxFunctionEvaluations - nf,
                "cycling_inner": cycling_inner,
                "with_cycling_memory": with_cycling_memory,
                "reduction_factor": reduction_factor,
                "forcing_function": forcing_function,
                "ftarget": ftarget,
                "polling_inner": polling_inner,
                "verbose": verbose
            }
            pdb.set_trace()
            # Perform the direct search in the block
            sub_xopt, sub_fopt, sub_exitflag, sub_output = inner_direct_search(
                fun, xbase, fbase, D[:, direction_indices], direction_indices,
                alpha_all[i_real], suboptions
            )

            # Record the index of the block visited
            num_visited_blocks += 1
            block_hist[num_visited_blocks - 1] = i_real
            
            # Record the step size used by inner_direct_search
            if output_alpha_hist:
                alpha_hist[:, iter - 1] = alpha_all
            
            # Record points visited if output_xhist is true
            if output_xhist:
                xhist[:, nf:nf + sub_output["nf"]] = sub_output["xhist"]
            
            # Record function values calculated by inner_direct_search
            fhist[nf:nf + sub_output["nf"]] = sub_output["fhist"]
            
            # Update the number of function evaluations
            nf += sub_output["nf"]
            
            # Update step size alpha_all based on reduction achieved
            if sub_fopt + reduction_factor[2] * forcing_function(alpha_all[i_real]) < fbase:
                alpha_all[i_real] *= expand
            elif sub_fopt + reduction_factor[1] * forcing_function(alpha_all[i_real]) >= fbase:
                alpha_all[i_real] *= shrink
            
            # Record best function value and point in block
            fopt_all[i_real] = sub_fopt
            xopt_all[:, i_real] = sub_xopt
            
            # Update xbase and fbase if not using "pads"
            if options["Algorithm"].lower() != "pads":
                if (reduction_factor[0] <= 0 and sub_fopt < fbase) or \
                   (sub_fopt + reduction_factor[0] * forcing_function(alpha_all[i_real]) < fbase):
                    xbase = sub_xopt
                    fbase = sub_fopt
            
            # Update direction indices for the next iteration
            direction_set_indices[i_real] = sub_output["direction_indices"]
            
            # Check termination conditions
            if sub_output["terminate"]:
                terminate = True
                exitflag = sub_exitflag
                break
            
            if np.max(alpha_all) < alpha_tol:
                terminate = True
                exitflag = get_exitflag("SMALL_ALPHA")
                break
        
        if terminate:
            break
        
        # Update xopt and fopt based on the best point found
        index = np.nanargmin(fopt_all)
        if fopt_all[index] < fopt:
            fopt = fopt_all[index]
            xopt = xopt_all[:, index]
        
        # For "pads", update xbase and fbase only after the outer loop
        if options["Algorithm"].lower() == "pads":
            if (reduction_factor[0] <= 0 and fopt < fbase) or \
               (fopt + reduction_factor[0] * forcing_function(np.min(alpha_all)) < fbase):
                xbase = xopt
                fbase = fopt

    # Record the number of function evaluations in output
    output["funcCount"] = nf

    # Truncate histories
    if output_block_hist:
        output["blocks_hist"] = block_hist[:num_visited_blocks]
    if output_alpha_hist:
        output["alpha_hist"] = alpha_hist[:, :min(iter, maxit)]
    if output_xhist:
        output["xhist"] = xhist[:, :nf]
    output["fhist"] = fhist[:nf]

    # Set message according to exitflag
    exit_messages = {
        get_exitflag("SMALL_ALPHA"): "The StepTolerance of the step size is reached.",
        get_exitflag("MAXFUN_REACHED"): "The maximum number of function evaluations is reached.",
        get_exitflag("FTARGET_REACHED"): "The target of the objective function is reached.",
        get_exitflag("MAXIT_REACHED"): "The maximum number of iterations is reached."
    }
    output["message"] = exit_messages.get(exitflag, "Unknown exitflag")

    # Transpose xopt if x0 is a row
    if x0_is_row:
        xopt = xopt.T
    
    # Verify postconditions if debug_flag is true
    if debug_flag:
        verify_postconditions(fun_orig, xopt, fopt, exitflag, output)

    return xopt, fopt, exitflag, output

bds(fun = chrosen, x0 = np.array([0, 0, 0]))