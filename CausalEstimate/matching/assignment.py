from typing import Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from CausalEstimate.matching.helpers import is_sparse


def assign_controls(distance_matrix: np.array) -> Tuple[np.array, np.array]:
    """
    Assigns controls to treated individuals using either sparse or dense assignment.
    Uses sparse assignment if the distance matrix is sparse; otherwise, falls back to dense assignment.
    """
    if is_sparse(distance_matrix):
        try:
            row_indices, col_indices = sparse_assignment(distance_matrix)
        except ValueError:
            raise ValueError("Cannot assign unique controls to treated.")
    else:
        distance_matrix[distance_matrix == 0] = np.inf
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

    return row_indices, col_indices


def sparse_assignment(dist_mat: np.array) -> Tuple[np.array, np.array]:
    """
    Performs assignment using sparse matching.
    """
    sparse_d_mat = csr_matrix(dist_mat)
    row_ind, col_ind = min_weight_full_bipartite_matching(sparse_d_mat)
    return row_ind, col_ind


def validate_control_availability(
    treated: pd.DataFrame, controls: pd.DataFrame, n_controls: int
) -> None:
    """
    Validates whether there are enough controls to match the treated individuals.
    Raises an error if there aren't enough controls.
    """
    if len(treated) == 0:
        raise ValueError("No treated units have sufficient controls")
    if len(controls) < n_controls * len(treated):
        raise ValueError(
            f"Not enough controls to match.\nN_controls: {len(controls)}\
            N_treated: {len(treated)}\
            Required N_controls: {n_controls*len(treated)} (n_controls x n_treated)"
        )
