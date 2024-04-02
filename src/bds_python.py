import numpy as np
import math
import pdb
from _isrow import is_row
from _verify_preconditions import verify_preconditions
from _ischarstr import ischarstr
from _get_default_constant import get_default_constant
from _get_direction_set import get_direction_set
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
        options["seed"] = get_default_constant("seed")
    np.random.seed(options["seed"])

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

    return result, debug_flag


options = {"debug_flag": True}
output1, output2 = bds(rosenbrock_example.chrosen, np.array([1, 2, 3, 4, 5]), options)
# output3 = output2.ndim <= 2 and output2.shape[-1] == 1
# print(output1, output2, output3, output4)
