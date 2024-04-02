from _isintegerscalar import isintegerscalar


def divide_direction_set(n, num_blocks, debug_flag):
    if debug_flag:
        if not isintegerscalar(n) or n <= 0:
            raise ValueError('n is not a positive integer.')
        if not isintegerscalar(num_blocks) or num_blocks <= 0:
            raise ValueError('num_blocks is not a positive integer.')
        if n < num_blocks:
            raise ValueError('The number of blocks should not be greater than the number of variables.')

    num_directions_each_block = n // num_blocks

    # num_directions_block indicates the number of directions of each block after averaging.
    if n % num_blocks == 0:
        num_directions_block = [num_directions_each_block] * num_blocks
    else:
        # The last block may have fewer directions than others.
        num_directions_block = [(num_directions_each_block + 1)] * (n % num_blocks) + [num_directions_each_block] * (
                    num_blocks - (n % num_blocks))

    # Use list instead of matrix in Python to avoid the number of directions in each block being different.
    index_direction_set = []
    for i in range(num_blocks):
        block_index_direction_set = list(range(sum(num_directions_block[:i]), sum(num_directions_block[:i + 1])))
        block_index_direction_set = [2 * x - 1 for x in block_index_direction_set]
        index_direction_set.append(block_index_direction_set)

    if debug_flag:
        if len(index_direction_set) != num_blocks:
            raise ValueError('The number of blocks of index_direction_set is not correct.')

    return index_direction_set
