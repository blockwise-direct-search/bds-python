import numpy as np
from _isrealvector import isrealvector
from _isintegerscalar import isintegerscalar
from _islogicalscalar import islogicalscalar


def cycling(array, index, strategy, with_cycling_memory, debug_flag):
    if debug_flag:

        if not isrealvector(array):
            raise ValueError("Array is not a real vector.")

        if not isintegerscalar(index):
            raise ValueError("Index is not an integer.")

        if not (isintegerscalar(strategy) and 0 <= strategy <= 4):
            raise ValueError("Strategy is not a positive integer or less than or equal to 4.")

        if not isintegerscalar(with_cycling_memory):
            raise ValueError("With_memory is not a boolean value.")

    if index < 0 or strategy == 0:
        return

    if not with_cycling_memory:
        array, indices = sorted(array)
        index = np.where(indices == index)[0]

    if strategy == 1:
        array[:index] = array[index - 1::-1]
    elif strategy == 2:
        array[index - 1:, :index - 1] = array[:index - 1, index:]
    elif strategy == 3:
        array[index:] = array[:index - 1]
    elif strategy == 4:
        if index != len(array):
            array[0:index + 1] = array[index::-1]

    if debug_flag:
        # Array should be a vector.
        if not isrealvector(array):
            raise ValueError("Array is not a real vector.")

    return array
