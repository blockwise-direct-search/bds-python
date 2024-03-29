import numpy as np

import ischarstr
import isrealvector
import isintegerscalar
import pdb


def check_row_vector(fun, x0, options):
    # VERIFY_PRECONDITIONS verifies the preconditions for the input arguments of the function.

    if not (ischarstr.ischarstr(fun) or callable(fun)):
        raise ValueError("fun should be a function handle.")

    if not isrealvector.isrealvector(x0)[0]:
        raise ValueError("x0 should be a real vector.")
    # pdb.set_trace()
    if "MaxFunctionEvaluations" in options:
        if not isintegerscalar.isintegerscalar(options["MaxFunctionEvaluations"]) and options["MaxFunctionEvaluations"] > 0:
            raise ValueError("MaxFunctionEvaluations should be an integer scalar.")


options = {"MaxFunctionEvaluations": 1000}
check_row_vector("abc", np.array([1, 2]), options)
