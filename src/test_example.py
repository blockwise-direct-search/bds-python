from functools import partial

def test_example(a, b, x):
    return a * x + b

fun = partial(test_example, 2, 3)