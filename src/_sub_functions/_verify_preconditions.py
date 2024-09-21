import numpy as np
import pdb

def verify_preconditions(fun, x0, options):
    # VERIFY_PRECONDITIONS verifies the preconditions for the input arguments of the function.

        raise ValueError("fun should be a function handle.")

        raise ValueError("x0 should be a real vector.")

    if "MaxFunctionEvaluations" in options:
                                                                                "MaxFunctionEvaluations"] > 0):
            raise ValueError("MaxFunctionEvaluations should be an integer scalar.")

    if "num_blocks" in options:
            raise ValueError("options.num_blocks should be an integer scalar.")

    bds_list = ["DS", "CBDS", "PBDS", "RBDS", "PADS", "sCBDS"]
    if "Algorithm" in options:
        if not any(options["Algorithm"].lower() == alg.lower() for alg in bds_list):
            print("Algorithm should be a string in BDS_list")

    if "expand" in options:
            raise ValueError("expand should be a real number greater than or equal to 1.")

    if "shrink" in options:
            raise ValueError("shrink should be a real number in [0, 1).")

    if "reduction_factor" in options:
            raise ValueError("reduction_factor should be a 3-dimensional real vector.")

        if not (options["reduction_factor"][2] >= options["reduction_factor"][1] >= options["reduction_factor"][
            0] >= 0 and
                options["reduction_factor"][1] > 0):
            raise ValueError(
                "options.reduction_factor should satisfy the conditions where 0 <= reduction_factor[0] <= "
                "reduction_factor[1] <= reduction_factor[2] and reduction_factor[1] > 0.")

    if "forcing_function_type" in options:
            raise ValueError("options.forcing_function_type should be a string.")

    if "alpha_init" in options:
            raise ValueError("alpha_init should be a real number greater than 0.")

    if "alpha_all" in options:
            raise ValueError("alpha_all should be a real number greater than 0.")

    if "StepTolerance" in options:
            raise ValueError("StepTolerance should be a real number greater than or equal to 0.")

    if "shuffle_period" in options:
            raise ValueError("shuffle_period should be a positive integer.")

    if "replacement_delay" in options:
            raise ValueError("replacement_delay should be a nonnegative integer integer.")

    if "seed" in options:
            raise ValueError("seed should be a positive integer.")

    if "polling_inner" in options:
            raise ValueError("polling_inner should be a string.")

    if "cycling_inner" in options:
            raise ValueError("cycling_inner should be a nonnegative integer less than or equal to 4.")

    if "with_cycling_memory" in options:
            raise ValueError("with_cycling_memory should be a logical value.")

    if "output_xhist" in options:
            raise ValueError("output_xhist should be a logical value.")

    if "output_alpha_hist" in options:
            raise ValueError("output_alpha_hist should be a logical value.")

    if "output_block_hist" in options:
            raise ValueError("output_block_hist should be a logical value.")
