import __init__


def divide_direction_set(n, num_blocks, debug_flag):
    if debug_flag:
        if not __init__.isintegerscalar(n) or n <= 0:
            raise ValueError('n is not a positive integer.')
        if not __init__.isintegerscalar(num_blocks) or num_blocks <= 0:
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

    index_direction_list = [[] for _ in range(num_blocks)]
    for i in range(1, num_blocks + 1):
        if i == 1:
            index_direction_list[i - 1] = list(range(0, 2*num_directions_block[i - 1], 1))
        else:
            index_direction_list[i - 1] = list(range(index_direction_list[i - 2][-1] + 1,
                                                     index_direction_list[i - 2][-1] + 1 + (
                                                                 2 * num_directions_block[i - 1] - 1) + 1))

    if debug_flag:
        if len(index_direction_list) != num_blocks:
            raise ValueError('The number of blocks of index_direction_set is not correct.')

    return index_direction_list

# result = divide_direction_set(5, 5, False)
# print(result)