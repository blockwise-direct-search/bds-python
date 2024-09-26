import pytest
import sys
import os

# Get the path of the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project root directory (project directory) to sys.path.
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_dir)
from sub_functions import *
from main import bds

def test_cycling():
    # CYCLING_TEST tests the file private/cycling.m

    # The following must not cycle the array.
    array = [1, 2, 3, 4, 5]
    for memory in [True, False]:
        for strategy in range(5):
            assert cycling(array, -1, strategy, memory) == array
        for index in range(len(array)):
            assert cycling(array, index, 0, memory) == array

    # The following must cycle the array.
    array = [1, 2, 3, 4, 5]
    for memory in [True, False]:
        assert np.array_equal(np.array(cycling(array, 2, 2, memory)), np.array([3, 4, 5, 1, 2]))
        assert np.array_equal(np.array(cycling(array, 2, 3, memory)), np.array([4, 5, 1, 2, 3]))
        assert np.array_equal(np.array(cycling(array, 2, 4, memory)), np.array([4, 1, 2, 3, 5]))

    array = [2, 1, 4, 5, 3]

    assert np.array_equal(np.array(cycling(array, 2, 1, True)), np.array([4, 2, 1, 5, 3]))
    assert np.array_equal(np.array(cycling(array, 2, 1, False)), np.array([4, 1, 2, 3, 5]))
    assert np.array_equal(np.array(cycling(array, 2, 2, True)), np.array([4, 5, 3, 2, 1]))
    assert np.array_equal(np.array(cycling(array, 2, 2, False)), np.array([4, 5, 1, 2, 3]))
    assert np.array_equal(np.array(cycling(array, 2, 3, True)), np.array([5, 3, 2, 1, 4]))
    assert np.array_equal(np.array(cycling(array, 2, 3, False)), np.array([5, 1, 2, 3, 4]))

def test_divide_direction_set():

    n = 11
    nb = 3
    INDEX_direction_set = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21]]
    assert divide_direction_set(n, nb) == INDEX_direction_set

    n = 10
    nb = 3
    INDEX_direction_set = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19]]
    assert divide_direction_set(n, nb) == INDEX_direction_set

    n = 15
    nb = 3
    INDEX_direction_set = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    assert divide_direction_set(n, nb) == INDEX_direction_set

    n = 3
    nb = 3
    INDEX_direction_set = [[0, 1], [2, 3], [4, 5]]
    assert divide_direction_set(n, nb) == INDEX_direction_set

def test_eval_fun():

    def eval_fun_tmp(x):
        if len(x) <= 100:
            return float('nan')
        elif len(x) <= 200:
            return float('inf')
        elif len(x) <= 300:
            return 2 ** 50
        else:
            raise ValueError('The length of x is too large.')

    n = np.random.randint(1, 100)
    x = np.random.randn(n)
    f_return = 2 ** 100
    f, f_real = eval_fun(eval_fun_tmp, x)
    assert f == f_return

    n = np.random.randint(101, 200)
    x = np.random.randn(n)
    f_return = 2 ** 100
    f, f_real = eval_fun(eval_fun_tmp, x)
    assert f == f_return

    n = np.random.randint(201, 300)
    x = np.random.randn(n)
    f_return = 2 ** 50
    f, f_real = eval_fun(eval_fun_tmp, x)
    assert f == f_return

def test_get_default_constant():
    assert get_default_constant("MaxFunctionEvaluations_dim_factor") == 500
    assert get_default_constant("Algorithm") == "cbds"
    assert get_default_constant("expand") == 2
    assert get_default_constant("shrink") == 0.5
    assert callable(get_default_constant("forcing_function"))
    assert get_default_constant("reduction_factor") == [0, np.finfo(float).eps, np.finfo(float).eps]
    assert get_default_constant("StepTolerance") == 1e-6
    assert get_default_constant("permuting_period") == 1
    assert get_default_constant("replacement_delay") == 1
    assert get_default_constant("ftarget") == -np.inf
    assert get_default_constant("polling_inner") == "opportunistic"
    assert get_default_constant("with_cycling_memory") == True
    assert get_default_constant("alpha_init") == 1
    assert get_default_constant("cycling_inner") == 1
    assert get_default_constant("output_xhist") == False
    assert get_default_constant("output_alpha_hist") == False
    assert get_default_constant("output_block_hist") == False
    assert get_default_constant("output_xhist_failed") == False
    assert get_default_constant("verbose") == False

def test_get_exitflag():
    assert get_exitflag("SMALL_ALPHA") == 0
    assert get_exitflag("FTARGET_REACHED") == 1
    assert get_exitflag("MAXFUN_REACHED") == 2
    assert get_exitflag("MAXIT_REACHED") == 3

def test_direction_set():
    n = 5
    D = np.zeros((n, 2 * n))
    for i in range(n):
        D[i, 2 * i] = 1
        D[i, 2 * i + 1] = -1
    assert(np.array_equal(get_direction_set(n), D))

    options = {}
    assert(np.array_equal(get_direction_set(n, options), D))

    n = 3
    options = {}
    A = np.random.randn(n, n)
    Q = qr(A)[0]
    options['direction_set'] = Q
    D = get_direction_set(n, options)
    assert(np.all(np.sort(D[:, 0:2 * n:2]) == np.sort(Q)))
    assert(np.all(D[:, 0::2] == -D[:, 1::2]))
    assert np.linalg.matrix_rank(D[:, 0::2]) == 3

    n = 3
    options = {}
    A = np.zeros((3, 3))
    detA = 0
    while detA == 0:
        A = np.random.randn(3, 3)
        detA = np.linalg.det(A)
    options['direction_set'] = A
    D = get_direction_set(n, options)
    assert(np.all(np.sort(D[:, 0:2 * n:2]) == np.sort(A)))
    assert (np.all(D[:, 0:2 * n:2] == -D[:, 1:2 * n:2]))
    assert np.linalg.matrix_rank(D[:, 0::2]) == n

    n = 5
    Q = qr(np.random.randn(n, n))[0]
    options['direction_set'] = Q
    D = get_direction_set(n, options)
    assert (np.all(np.sort(D[:, 0:2 * n:2]) == np.sort(Q)))
    assert(np.all(D[:, 0:2 * n:2] == -D[:, 1:2 * n:2]))
    assert np.linalg.matrix_rank(D[:, 0:2 * n:2]) == n

    n = 5
    options['direction_set'] = np.full((n, n), np.nan)
    D = np.zeros((n, 2 * n))
    for i in range(n):
        D[i, 2 * i] = 1
        D[i, 2 * i + 1] = -1
    assert(np.array_equal(get_direction_set(n, options), D))

    n = 5
    options['direction_set'] = np.full((n, n), np.inf)
    D = np.zeros((n, 2 * n))
    for i in range(n):
        D[i, 2 * i] = 1
        D[i, 2 * i + 1] = -1
    assert(np.array_equal(get_direction_set(n, options), D))

if __name__ == '__main__':
    pytest.main([__file__])
