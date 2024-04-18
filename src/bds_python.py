import __init__
from examples.rosenbrock_example import chrosen


# To my understanding, it is more reasonable to let x0 be a numpy vector in our algorithm, not list.
# Thus, we need to convert x0 to a numpy vector if it is a list.
def bds(fun, x0, options=None):
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

    # If options is None, then set it to an empty dictionary.
    if options is None:
        options = {}

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


output1 = bds(chrosen, [1, 2, 3, 4, 5], options={"verbose": "True"})
# output3 = output2.ndim <= 2 and output2.shape[-1] == 1
# print(output1, output2, output3, output4)
