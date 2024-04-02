import numpy as np

from _ischarstr import ischarstr
from _isrealvector import isrealvector
from _isrealscalar import isrealscalar
from _isintegerscalar import isintegerscalar
from _isnumvec import isnumvec
from _islogicalscalar import islogicalscalar
import pdb


def verify_preconditions(fun, x0, options):
    # VERIFY_PRECONDITIONS verifies the preconditions for the input arguments of the function.

    if not (ischarstr(fun) or callable(fun)):
        raise ValueError("fun should be a function handle.")

    if not isrealvector(x0)[0]:
        raise ValueError("x0 should be a real vector.")
    # pdb.set_trace()
    if "MaxFunctionEvaluations" in options:
        if not (isintegerscalar(options["MaxFunctionEvaluations"]) and options[
                                                                                "MaxFunctionEvaluations"] > 0):
            raise ValueError("MaxFunctionEvaluations should be an integer scalar.")

    if "num_blocks" in options:
        if not (isintegerscalar(options["num_blocks"]) and options["num_blocks"] > 0):
            raise ValueError("options.num_blocks should be an integer scalar.")

    bds_list = ["DS", "CBDS", "PBDS", "RBDS", "PADS", "sCBDS"]
    if "Algorithm" in options:
        if not any(options["Algorithm"].lower() == alg.lower() for alg in bds_list):
            print("Algorithm should be a string in BDS_list")

    if "expand" in options:
        if not (isrealscalar(options["expand"]) and options["expand"] >= 1):
            raise ValueError("expand should be a real number greater than or equal to 1.")

    if "shrink" in options:
        if not (isrealscalar(options["shrink"]) and options["expand"] > 0 and options["shrink"] < 1):
            raise ValueError("shrink should be a real number in [0, 1).")

    if "reduction_factor" in options:
        if not (isnumvec(options["reduction_factor"]) and len(options["reduction_factor"]) == 3):
            raise ValueError("reduction_factor should be a 3-dimensional real vector.")

        if not (options["reduction_factor"][2] >= options["reduction_factor"][1] >= options["reduction_factor"][
            0] >= 0 and
                options["reduction_factor"][1] > 0):
            raise ValueError(
                "options.reduction_factor should satisfy the conditions where 0 <= reduction_factor[0] <= "
                "reduction_factor[1] <= reduction_factor[2] and reduction_factor[1] > 0.")

    if "forcing_function_type" in options:
        if not ischarstr(options["forcing_function_type"]):
            raise ValueError("options.forcing_function_type should be a string.")

    if "alpha_init" in options:
        if not (isrealscalar(options["alpha_init"]) and options["alpha_init"] > 0):
            raise ValueError("alpha_init should be a real number greater than 0.")

    if "alpha_all" in options:
        if not (isrealscalar(options["alpha_all"]) and options["alpha_all"] > 0):
            raise ValueError("alpha_all should be a real number greater than 0.")

    if "StepTolerance" in options:
        if not (isrealscalar(options["StepTolerance"]) and options["StepTolerance"] >= 0):
            raise ValueError("StepTolerance should be a real number greater than or equal to 0.")

    if "shuffle_period" in options:
        if not (isintegerscalar(options["shuffle_period"]) and options["shuffle_period"] > 0):
            raise ValueError("shuffle_period should be a positive integer.")

    if "replacement_delay" in options:
        if not (isintegerscalar(options["replacement_delay"]) and options["replacement_delay"] >= 0):
            raise ValueError("replacement_delay should be a nonnegative integer integer.")

    if "seed" in options:
        if not (isintegerscalar(options["seed"]) and options["seed"] > 0):
            raise ValueError("seed should be a positive integer.")

    if "polling_inner" in options:
        if not (ischarstr(options["polling_inner"])):
            raise ValueError("polling_inner should be a string.")

    if "cycling_inner" in options:
        if not (isintegerscalar(options["cycling_inner"]) and 0 <= options["cycling_inner"] <= 4):
            raise ValueError("cycling_inner should be a nonnegative integer less than or equal to 4.")

    if "with_cycling_memory" in options:
        if not (islogicalscalar(options["with_cycling_memory"])):
            raise ValueError("with_cycling_memory should be a logical value.")

    if "output_xhist" in options:
        if not (islogicalscalar(options["output_xhist"])):
            raise ValueError("output_xhist should be a logical value.")

    if "output_alpha_hist" in options:
        if not (islogicalscalar(options["output_alpha_hist"])):
            raise ValueError("output_alpha_hist should be a logical value.")

    if "output_block_hist" in options:
        if not (islogicalscalar(options["output_block_hist"])):
            raise ValueError("output_block_hist should be a logical value.")
