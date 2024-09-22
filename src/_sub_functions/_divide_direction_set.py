import numpy as np

def divide_direction_set(n, num_blocks, debug_flag):

    r"""
    DIVIDE_DIRECTION_SET gets indices of the directions in each block.
    INDEX_DIRECTION_SET = DIVIDE_DIRECTION_SET(n, num_blocks) returns a cell, 
    where the index_direction_set{i} contains the indices of the directions in 
    the i-th block. In our implementation, the direction set is in the form of 
    [d_1, -d_1, d_2, -d_2, ..., d_n, -d_n], containing 2n directions. 
    We divide the direction set into num_blocks blocks, with the first 
    mod(n, num_blocks) blocks containing 2*(floor(n/num_blocks) + 1) directions 
    and the rest containing 2*floor(n/num_blocks) directions. 

    Example
    n = 11, num_blocks = 3.
    The number of directions in each block is 4*2, 4*2, 3*2 respectively.
    Thus INDEX_DIRECTION_SET is a list, where
    index_direction_set[1] = [1, 2, 3, 4, 5, 6, 7, 8],
    index_direction_set[2] = [9, 10, 11, 12, 13, 14, 15, 16],
    index_direction_set[3] = [17, 18, 19, 20, 21, 22].
    """

    if debug_flag:
        if not (isinstance(n, int) or n <= 0):
            raise ValueError('n is not a positive integer.')
        if not (isinstance(num_blocks, int) or num_blocks <= 0):
            raise ValueError('num_blocks is not a positive integer.')
        if n < num_blocks:
            raise ValueError('The number of blocks should not be greater than the number of variables.')


    # Calculate the number of directions of each block.
    # We try to make the number of directions in each block as even as possible.
    # In specific, the first mod(n, num_blocks) blocks contain 2*(floor(n/num_blocks) + 1)
    # directions and the rest contain 2*floor(n/num_blocks) directions.
    num_directions_block = np.ones(num_blocks) * (n // num_blocks)
    num_directions_block[:n % num_blocks] += 1
    num_directions_block *= 2
    # Since we use np.ones to initialize the array, the type of num_directions_block
    # is float. We need to convert it to int.
    num_directions_block = num_directions_block.astype(int)

    index_direction_set = [None] * num_blocks
    # Since we use np.cumsum to calculate the initial index of each block, the type of
    # initial_index_each_block is float. We need to convert it to int.
    initial_index_each_block = (np.cumsum(np.concatenate(([1], num_directions_block[:-1])))).astype(int)

    for i in range(num_blocks):
        index_direction_set[i] = list(range(initial_index_each_block[i], initial_index_each_block[i] + num_directions_block[i]))

    if debug_flag:
        if len(index_direction_set) != num_blocks:
            raise ValueError('The number of blocks of index_direction_set is not correct.')

    return index_direction_set