import sys


def get_default_constant(constant_name):
    if constant_name == "MaxFunctionEvaluations_dim_factor":
        constant_value = 500
    elif constant_name == "Algorithm":
        constant_value = "cbds"
    elif constant_name == "expand":
        constant_value = 1.5
    elif constant_name == "shrink":
        constant_value = 0.4
    elif constant_name == "reduction_factor":
        constant_value = [0, sys.float_info.min, sys.float_info.min]
    elif constant_name == "forcing_function":
        constant_value = lambda alpha: alpha ** 2
    elif constant_name == "alpha_init":
        constant_value = 1
    elif constant_name == "StepTolerance":
        constant_value = 1e-6
    elif constant_name == "permuting_period":
        constant_value = 1
    elif constant_name == "replacement_delay":
        constant_value = 1
    elif constant_name == "ftarget":
        constant_value = -float("inf")
    elif constant_name == "polling_inner":
        constant_value = "opportunistic"
    elif constant_name == "cycling_inner":
        constant_value = 1
    elif constant_name == "with_cycling_memory":
        constant_value = True
    elif constant_name == "output_xhist":
        constant_value = False
    elif constant_name == "output_alpha_hist":
        constant_value = False
    elif constant_name == "output_block_hist":
        constant_value = False
    elif constant_name == "iprint":
        constant_value = 0
    else:
        raise ValueError("Unknown constant name")

    return constant_value
