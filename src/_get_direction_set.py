import __init__


def get_direction_set(n, options=None):
    if options is None:
        options = {}

    if not "direction_set" in options:
        D = __init__.np.full((n, 2 * n), __init__.np.nan)
        D[:, ::2] = __init__.np.eye(n)
        D[:, 1::2] = -__init__.np.eye(n)
    else:
        direction_set = options["direction_set"]
        if __init__.np.any(__init__.np.isnan(direction_set) or __init__.np.isinf(direction_set)):
            __init__.warnings.warn("direction_set contains NaN or Inf.")
            direction_set[__init__.np.isnan(direction_set) | __init__.np.isinf(direction_set)] = 0

        if __init__.np.abs(__init__.np.linalg.matrix_rank(direction_set) - direction_set.shape[1]) >= 1.0e-10:
            raise ValueError("direction_set is not linear independent.")

        shortest_direction_norm = 10 * __init__.np.sqrt(n) * __init__.np.finfo(float).eps
        direction_norms = __init__.np.linalg.norm(direction_set, axis=0)
        short_directions = direction_norms < shortest_direction_norm

        if __init__.np.any(short_directions):
            warning_message = ("The direction set contains directions shorter than %g. They are removed."
                               % shortest_direction_norm)
            __init__.warnings.warn(warning_message)
            direction_set = direction_set[:, ~short_directions]
            direction_norms = direction_norms[~short_directions]

        parallel_directions = __init__.np.abs(__init__.np.transpose(direction_set) @ direction_set) > (1 - 1.0e-10) * (
            __init__.np.outer(direction_norms, direction_norms))
        parallel_direction_indices = __init__.np.where(__init__.np.triu(parallel_directions, k=1))
        parallel_direction_indices = parallel_direction_indices[1]
        preserved_indices = __init__.np.setdiff1d(__init__.np.arange(1, direction_set.shape[1] + 1), parallel_direction_indices)
        direction_set = direction_set[:, preserved_indices - 1]

        if direction_set.size == 0:
            direction_set = __init__.np.eye(n)

        Q, R = __init__.np.linalg.qr(direction_set)
        _, m = direction_set.shape
        deficient_columns = (~(__init__.np.abs(__init__.np.diag(R[0:min(m, n), 0:min(m, n)]))) > 10 * __init__.np.finfo(float).eps * max(m,
                                                                                                              n)
                             * __init__.np.linalg.norm(
                    R[0:min(m, n), 0:min(m, n)]))

        direction_set = __init__.np.concatenate((direction_set, Q[:, deficient_columns], Q[:, m:]), axis=1)

        m = direction_set.shape[1]
        D = __init__.np.full((n, 2 * m), __init__.np.nan)
        D[:, 0:2:2 * m - 1] = direction_set
        D[:, 1:2:2 * m] = -direction_set

    return D
