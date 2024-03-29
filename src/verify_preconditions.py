import ischarstr
import isrealvector
import pdb

def check_row_vector(fun, x0, options):
    # VERIFY_PRECONDITIONS verifies the preconditions for the input arguments of the function.

    if not (ischarstr.ischarstr(fun) or callable(fun)):
        raise ValueError("fun should be a function handle.")

    if not isrealvector.isrealvector(x0)[0]:
        raise ValueError("x0 should be a real vector.")
    pdb.set_trace()

check_row_vector("abc", "def", "def")