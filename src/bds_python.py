import numpy as np
import pdb
import isrow
import verify_preconditions



def bds(var1, x0, options=None):
    if options is None:
        options = {}

    result1 = var1

    # Transpose x0 if it is a row vector.
    x0_is_row = isrow.check_row_vector(x0)
    if x0_is_row:
        x0 = x0.reshape(-1, 1)
    #pdb.set_trace()
    result2 = x0

    # Set the default value of debug_flag. If options do not contain debug_flag, then
    # debug_flag is set to false.
    if "debug_flag" in options:
        debug_flag = options["debug_flag"]
    else:
        debug_flag = False

    if debug_flag:
        print("x0_is_row: ", x0_is_row)

    return result1, result2, debug_flag


output1, output2, output4 = bds(1, np.array([1, 2, 3, 4, 5]))
output3 = output2.ndim <= 2 and output2.shape[-1] == 1
print(output1, output2, output3, output4)
