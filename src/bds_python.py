import numpy as np
import math
import pdb
from _isrow import is_row
from _verify_preconditions import verify_preconditions
from _ischarstr import ischarstr
from _get_default_constant import get_default_constant
from _get_direction_set import get_direction_set
from _divide_direction_set import divide_direction_set
from _get_exitflag import get_exitflag
from _eval_fun import eval_fun
from _inner_direct_search import inner_direct_search
import functools
from examples import rosenbrock_example


# To my understanding, it is more reasonable to let x0 be a numpy vector, not list.
def bds(fun, x0, options=None):
    if options is None:
        options = {}

    # Transpose x0 if it is a row vector.
    x0_is_row = is_row(x0)
    if x0_is_row:
        x0 = x0.reshape(-1, 1)
    # pdb.set_trace()
    result = x0

    # Set the default value of debug_flag. If options do not contain debug_flag, then
    # debug_flag is set to false.
    if "debug_flag" in options:
        debug_flag = options["debug_flag"]
    else:
        debug_flag = False

    if debug_flag:
        print("x0_is_row: ", x0_is_row)
        verify_preconditions(fun, x0, options)

    # If FUN is a string, then convert it to a function handle.
    if ischarstr(fun):
        fun = eval(fun)
    # Redefine fun to accept column vectors if x0 is a row vector, as we use column vectors internally.
    fun_orig = fun
    if x0_is_row:
        fun = functools.partial(fun, x=lambda x: x.T)

    if not "seed" in options:
        options["seed"] = list(range(624))
    random_stream = np.random.default_rng()
    random_stream.shuffle(options["seed"])

    n = len(x0)
    if not "Algorithm" in options:
        options["Algorithm"] = get_default_constant("Algorithm")

    if "direction_set_type" in options:
        if options["direction_set_type"].lower() == "randomized_orthogonal":
            random_matrix = np.random.randn(n * n).reshape(n, n)
            options["direction_set"], _ = np.linalg.qr(random_matrix)
        elif options["direction_set_type"].lower() == "randomized":
            options["direction_set"] = np.random.randn(n, n)

    D = get_direction_set(n, options)

    num_directions = D.shape[1]
    if options["Algorithm"].lower() == "ds":
        num_blocks = 1
    elif "block" in options:
        num_blocks = min(num_directions, options["block"])
    elif options["Algorithm"].lower() in ["cbds", "pbds", "rbds", "pads", "scbds"]:
        num_blocks = math.ceil(num_directions / 2)
    else:
        raise ValueError("The field Algorithm of options is invalid.")

    direction_set_indices = divide_direction_set(n, num_blocks, debug_flag)

    if "expand" in options:
        expand = options["expand"]
    else:
        expand = get_default_constant("expand")

    if "shrink" in options:
        shrink = options["shrink"]
    else:
        shrink = get_default_constant("shrink")

    if "reduction_factor" in options:
        reduction_factor = options["reduction_factor"]
    else:
        reduction_factor = get_default_constant("reduction_factor")

    if "forcing_function" in options:
        forcing_function = options["forcing_function"]
    else:
        forcing_function = get_default_constant("forcing_function")
    if "forcing_function_type" in options:
        if options["forcing_function_type"] == "quadratic":
            forcing_function = lambda x: x ** 2
        elif options["forcing_function_type"] == "cubic":
            forcing_function = lambda x: x ** 3

    if "polling_inner" in options:
        polling_inner = options["polling_inner"]
    else:
        polling_inner = get_default_constant("polling_inner")

    if "cycling_inner" in options:
        cycling_inner = options["cycling_inner"]
    else:
        cycling_inner = get_default_constant("cycling_inner")

    if options["Algorithm"].lower() == "pbds":
        if "permuting_period" in options:
            permuting_period = options["permuting_period"]

    if options["Algorithm"].lower() == "rbds":
        if "replacement_delay" in options:
            replacement_delay = min(options["replacement_delay"], num_blocks - 1)
        else:
            replacement_delay = min(get_default_constant("replacement_delay"), num_blocks - 1)

    if "with_cycling_memory" in options:
        with_cycling_memory = options["with_cycling_memory"]
    else:
        with_cycling_memory = get_default_constant("with_cycling_memory")

    if "MaxFunctionEvaluations" in options:
        MaxFunctionEvaluations = options["MaxFunctionEvaluations"]
    else:
        MaxFunctionEvaluations = get_default_constant("MaxFunctionEvaluations_dim_factor") * n

    maxit = MaxFunctionEvaluations

    if "StepTolerance" in options:
        alpha_tol = options["StepTolerance"]
    else:
        alpha_tol = get_default_constant("StepTolerance")

    if "ftarget" in options:
        ftarget = options["ftarget"]
    else:
        ftarget = get_default_constant("ftarget")

    if "output_alpha_hist" in options:
        output_alpha_hist = options["output_alpha_hist"]
    else:
        output_alpha_hist = get_default_constant("output_alpha_hist")
    if output_alpha_hist:
        try:
            alpha_hist = np.full((num_blocks, maxit), np.nan)
        except:
            output_alpha_hist = False
            print("alpha_hist will not be included in the output due to memory limitations.")

    if "alpha_init" in options:
        if len(options["alpha_init"]) == 1:
            alpha_all = options["alpha_init"] * np.ones((num_blocks, 1))
        elif len(options["alpha_init"]) == num_blocks:
            alpha_all = options["alpha_init"]
        else:
            raise ValueError("The length of alpha_init should be equal to num_blocks or equal to 1.")
    elif num_blocks == n and D.shape[1] == 2 * n and "alpha_init_scaling" in options and options["alpha_init_scaling"]:
        x0_coordinates = np.linalg.lstsq(D[:, 0:2:2 * n - 1], x0, rcond=None)[0]
        x0_scales = np.abs(x0_coordinates)
        if "alpha_init_scaling_factor" in options:
            alpha_all = options["alpha_init_scaling_factor"] * x0_scales
        else:
            alpha_all = 0.5 * np.maximum(1, np.abs(x0_scales))
    else:
        alpha_all = np.ones((num_blocks, 1))

    fopt_all = np.full((1, num_blocks), np.nan)
    xopt_all = np.full((n, num_blocks), np.nan)

    fhist = np.full((1, MaxFunctionEvaluations), np.nan)

    if "output_xhist" in options:
        output_xhist = options["output_xhist"]
    else:
        output_xhist = get_default_constant("output_xhist")
    if output_xhist:
        try:
            xhist = np.full((n, maxit), np.nan)
        except:
            output_xhist = False
            print("xhist will not be included in the output due to memory limitations.")

    if "output_block_hist" in options:
        output_block_hist = options["output_block_hist"]
    else:
        output_block_hist = get_default_constant("output_block_hist")
    block_hist = np.full((1, maxit), np.nan)

    if "iprint" in options:
        iprint = options["iprint"]
    else:
        iprint = get_default_constant("iprint")

    exitflag = get_exitflag("MAXIT_REACHED", debug_flag)

    xbase = x0
    [fbase, fbase_real] = eval_fun(fun, xbase)
    if iprint == 1:
        print("Function number %d, F = %f" % (1, fbase))
        print("The corresponding X is:")
        print(" ".join(map(str, xbase.flatten())))
        print()

    xopt = xbase
    fopt = fbase

    nf = 1
    if output_xhist:
        xhist[:, nf - 1] = xbase

    fhist[nf - 1] = fbase_real

    terminate = False
    if fopt <= ftarget:
        information = "FTARGET_REACHED"
        exitflag = get_exitflag(information, debug_flag)
        maxit = 0

    if "block_indices_permuted_init" in options and options["block_indices_permuted_init"]:
        all_block_indices = np.random.permutation(num_blocks)
    else:
        all_block_indices = np.arange(1, num_blocks + 1)
    num_visited_blocks = 0

    for iteration in range(1, maxit + 1):

        if options["Algorithm"].lower() in ["ds", "cbds", "pads"]:
            block_indices = all_block_indices
        elif options["Algorithm"].lower() == "pbds" and (iteration - 1) % permuting_period == 0:
            block_indices = np.random.permutation(num_blocks)
        elif options["Algorithm"].lower() == "rbds":
            unavailable_block_indices = block_hist[max(1, iteration - replacement_delay): iteration - 1]
            available_block_indices = np.setdiff1d(all_block_indices, unavailable_block_indices)
            idx = np.random.randint(len(available_block_indices))
            block_indices = available_block_indices[idx]  # a vector of length 1
        elif options["Algorithm"].lower() == "scbds":
            block_indices = np.concatenate([all_block_indices, np.arange(num_blocks - 1, 1, -1)])

        for i in range(1, len(block_indices) + 1):
            i_real = block_indices[i - 1] - 1

            direction_indices = direction_set_indices[i_real]

            suboptions = {"FunctionEvaluations_exhausted": nf, "MaxFunctionEvaluations": MaxFunctionEvaluations - nf,
                          "cycling_inner": cycling_inner, "with_cycling_memory": with_cycling_memory,
                          "reduction_factor": reduction_factor, "forcing_function": forcing_function,
                          "ftarget": ftarget, "polling_inner": "polling_inner", "iprint": iprint,
                          "debug_flag": debug_flag}

            [sub_xopt, sub_fopt, sub_exitflag, sub_output] = inner_direct_search(fun, xbase,
                                                                                 fbase, D[:, direction_indices],
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
                exitflag = get_exitflag("SMALL_ALPHA", debug_flag)
                break

        _, index = np.nanmin(fopt_all)
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
    if exitflag_value == get_exitflag("SMALL_ALPHA", debug_flag):
        output["message"] = "The StepTolerance of the step size is reached."
    elif exitflag_value == get_exitflag("MAXFUN_REACHED", debug_flag):
        output["message"] = "The maximum number of function evaluations is reached."
    elif exitflag_value == get_exitflag("FTARGET_REACHED", debug_flag):
        output["message"] = "The target of the objective function is reached."
    elif exitflag_value == get_exitflag("MAXIT_REACHED", debug_flag):
        output["message"] = "The maximum number of iterations is reached."
    else:
        output["message"] = "Unknown exitflag"

    if x0_is_row:
        xopt = xopt.T

    return xopt, fopt, exitflag, output


options = {"debug_flag": True}
output1, output2 = bds(rosenbrock_example.chrosen, np.array([1, 2, 3, 4, 5]), options)
# output3 = output2.ndim <= 2 and output2.shape[-1] == 1
# print(output1, output2, output3, output4)
