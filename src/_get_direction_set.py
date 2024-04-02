import numpy as np
import warnings


def get_direction_set(n, options=None):
    if options is None:
        options = {}

    if not "direction_set" in options:
        D = np.full((n, 2 * n), np.nan)
        D[:, ::2] = np.eye(n)
        D[:, 1::2] = -np.eye(n)
    else:
        direction_set = options["direction_set"]
        if np.any(np.isnan(direction_set) or np.isinf(direction_set)):
            warnings.warn("direction_set contains NaN or Inf.")
            direction_set[np.isnan(direction_set) | np.isinf(direction_set)] = 0

        if np.abs(np.linalg.matrix_rank(direction_set) - direction_set.shape[1]) >= 1.0e-10:
            raise ValueError("direction_set is not linear independent.")

        shortest_direction_norm = 10 * np.sqrt(n) * np.finfo(float).eps
        direction_norms = np.linalg.norm(direction_set, axis=0)
        short_directions = direction_norms < shortest_direction_norm

        if np.any(short_directions):
            warning_message = ("The direction set contains directions shorter than %g. They are removed."
                               % shortest_direction_norm)
            warnings.warn(warning_message)
            direction_set = direction_set[:, ~short_directions]
            direction_norms = direction_norms[~short_directions]

        parallel_directions = np.abs(np.transpose(direction_set) @ direction_set) > (1 - 1.0e-10) * (
            np.outer(direction_norms, direction_norms))
        parallel_direction_indices = np.where(np.triu(parallel_directions, k=1))
        parallel_direction_indices = parallel_direction_indices[1]
        preserved_indices = np.setdiff1d(np.arange(1, direction_set.shape[1] + 1), parallel_direction_indices)
        direction_set = direction_set[:, preserved_indices - 1]

        if direction_set.size == 0:
            direction_set = np.eye(n)

        Q, R = np.linalg.qr(direction_set)
        _, m = direction_set.shape
        deficient_columns = (~(np.abs(np.diag(R[0:min(m, n), 0:min(m, n)]))) > 10 * np.finfo(float).eps * max(m,
                                                                                                              n)
                             * np.linalg.norm(
                    R[0:min(m, n), 0:min(m, n)]))

        direction_set = np.concatenate((direction_set, Q[:, deficient_columns], Q[:, m:]), axis=1)

        m = direction_set.shape[1]
        D = np.full((n, 2 * m), np.nan)
        D[:, 0:2:2 * m - 1] = direction_set
        D[:, 1:2:2 * m] = -direction_set

    return D
