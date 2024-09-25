import numpy as np
from ._isrealvector import isrealvector

def cycling(array, index, strategy, with_cycling_memory, debug_flag=False):
    """
    Permutes an array according to different options.

    Args:
        array: The array to permute. Must be a vector.
        index: An integer indicating the permutation index.
        strategy: A nonnegative integer indicating the strategy of the permutation (0 to 4).
        with_cycling_memory: A boolean indicating whether to use cycling memory.

    Returns:
        array: The permuted array.
    """

    # Check input types if debugging is enabled
    if debug_flag:
        if not isrealvector(array):
            raise ValueError("Array is not a real vector.")
        if not isinstance(index, int):
            raise ValueError("Index is not an integer.")
        if not (isinstance(strategy, int) and 0 <= strategy <= 4):
            raise ValueError("Strategy is not a valid integer (0 to 4).")
        if not isinstance(with_cycling_memory, bool):
            raise ValueError("With_memory is not a boolean value.")

    # No permutation if index < 0 or strategy == 0
    if index < 0 or strategy == 0:
        return array

    # If with_cycling_memory is false, sort the array and adjust the index
    if not with_cycling_memory:
        array_orig = array
        array = np.sort(array)
        index = np.where(array == array_orig[index])[0][0]

    # Perform cycling based on the strategy
    if strategy == 1:
        # Move the element at index to the front
        array = np.concatenate(([array[index]], np.delete(array, index)))
    elif strategy == 2:
        # Move the element at index and following elements to the front
        array = np.concatenate((array[index:], array[:index]))
    elif strategy == 3:
        # Move the following elements after the index to the front
        array = np.concatenate((array[index + 1:], array[:index + 1]))
    elif strategy == 4:
        # Move the next element after the index to the front
        if index != len(array) - 1:
            array = np.concatenate(([array[index + 1]], array[:index + 1], array[index + 2:]))

    # Check whether the array is a vector after cycling if debugging is enabled
    if debug_flag:
        if not isrealvector(array):
            raise ValueError("Array is not a real vector.")

    return array