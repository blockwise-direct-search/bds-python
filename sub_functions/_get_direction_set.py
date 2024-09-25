import numpy as np
from scipy.linalg import qr
import pdb


def get_direction_set(n, options=None):
    """
    Generates the set of polling directions.

    If the user does not input options or the options do not contain the field of direction_set,
    D will be [e_1, -e_1, ..., e_n, -e_n], where e_i is the i-th coordinate vector.
    Otherwise, options.direction_set should be an n-by-n nonsingular matrix, and D will be
    set to [d_1, -d_1, ..., d_n, -d_n], where d_i is the i-th column in options.direction_set.
    If the columns of options.direction_set are almost linearly dependent, then we will revise direction_set
    in the following way.
    1. Remove the directions whose norms are too small.
    2. Find directions that are almost parallel. Then preserve the first one and remove the others.
    3. Find a maximal linearly independent subset of the directions, and supplement this subset with
       new directions to make a basis of the full space. The final direction set will be this basis.
       This is done by QR factorization.
    """

    # Set options to an empty dictionary if it is not provided.
    if options is None:
        options = {}

    if 'direction_set' not in options:

        # Set the default direction set.
        direction_set = np.eye(n)

    else:
        direction_set = options['direction_set']

        # Replace NaN or Inf values with 0.
        if np.any(np.isnan(direction_set)) or np.any(np.isinf(direction_set)):
            print("Warning: Some directions contain NaN or inf, which are replaced with 0.")
            direction_set[np.isnan(direction_set) | np.isinf(direction_set)] = 0

        # Remove directions whose norms are too small.
        shortest_direction_norm = 10 * np.sqrt(n) * np.finfo(float).eps
        direction_norms = np.linalg.norm(direction_set, axis=0)
        short_directions = direction_norms < shortest_direction_norm

        # If the direction set is empty, set it to an identity matrix.
        if np.any(short_directions) and direction_set.size > 0:
            print(
                f"Warning: The direction set contains directions shorter than {shortest_direction_norm}. They are removed.")
            direction_set = direction_set[:, ~short_directions]
            direction_norms = direction_norms[~short_directions]

        # Find directions that are almost parallel and remove duplicates.
        parallel_directions = np.abs(direction_set.T @ direction_set) > (1 - 1.0e-10) * (
                    direction_norms.T @ direction_norms)
        parallel_direction_indices = np.triu(parallel_directions, 1).nonzero()[1]

        # Remove the duplicate indices.
        preserved_indices = np.setdiff1d(np.arange(direction_set.shape[1]), parallel_direction_indices)
        direction_set = direction_set[:, preserved_indices]

        # If the direction set is empty, set it to the identity matrix.
        if direction_set.size == 0:
            direction_set = np.eye(n)
        else:
            # QR factorization to find a maximal linearly independent subset.
            Q, R, p = qr(A, pivoting=True)  # qr factorization with column pivoting.
            num_directions = direction_set.shape[1]
            is_independent = np.abs(np.diag(R)) >= 1e-10
            direction_set = np.hstack(
                (direction_set[:, p[is_independent]], Q[:, p[~is_independent]], Q[:, num_directions:n]))

    # Finally, set D to [d_1, -d_1, ..., d_n, -d_n].
    D = np.full((n, 2 * n), np.nan)
    D[:, 0:2*n:2] = direction_set
    D[:, 1:2*n:2] = -direction_set

    return D