import numpy as np

def get_default_constant(constant_name):
    r'''
    GET_DEFAULT_CONSTANT gets the default value of OPTIONS for BDS.
    '''
    if constant_name == "MaxFunctionEvaluations_dim_factor":
        constant_value = 500
    elif constant_name == "Algorithm":
        constant_value = "cbds"
    elif constant_name == "expand":
        constant_value = 2
    elif constant_name == "shrink":
        constant_value = 0.5
    elif constant_name == "reduction_factor":
        constant_value = [0, np.finfo(float).eps, np.finfo(float).eps]
    elif constant_name == "forcing_function":
        constant_value = lambda x: x ** 2
    elif constant_name == "alpha_init":
        constant_value = 1
    elif constant_name == "StepTolerance":
        constant_value = 1e-6
    elif constant_name == "permuting_period":
        constant_value = 1
    elif constant_name == "replacement_delay":
        constant_value = 1
    elif constant_name == "ftarget":
        constant_value = -np.inf
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
    elif constant_name == "output_xhist_failed":
        constant_value = False
    elif constant_name == "verbose":
        constant_value = False
    else:
        raise ValueError("Unknown constant name")

    return constant_value