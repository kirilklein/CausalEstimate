from typing import Tuple

import numpy as np
import pandas as pd


def compute_distance_matrix(
    treated: pd.DataFrame, control: pd.DataFrame, ps_col: str
) -> np.array:
    """
    Computes the distance matrix (absolute differences) between treated and control individuals based on propensity scores.
    Args:
        treated_df (pd.DataFrame): DataFrame of treated individuals.
        control_df (pd.DataFrame): DataFrame of control individuals.
        ps_col (str): Column name for propensity scores.
    Returns:
        np.array: Distance matrix with treated individuals as rows and controls as columns.
    """
    treated_ps = treated[ps_col].to_numpy()
    control_ps = control[ps_col].to_numpy()
    dist_mat = np.abs(treated_ps.reshape(-1, 1) - control_ps.reshape(1, -1))
    return dist_mat


def filter_treated_w_insufficient_controls(
    dist_mat: np.array, treated_df: pd.DataFrame, n_controls: int
) -> Tuple[np.array, pd.DataFrame]:
    """
    Filters out treated individuals who do not have enough valid controls based on the distance matrix.
    Args:
        distance_matrix (np.array): Distance matrix with treated individuals as rows and controls as columns.
                            zero entries indicate invalid matches.
        treated_df (pd.DataFrame): DataFrame of treated individuals.
        n_controls (int): Minimum number of controls required for each treated individual.
    Returns:
        Tuple[np.array, pd.DataFrame]: Updated distance matrix and treated DataFrame after filtering.
    """
    valid_controls_per_treated = np.count_nonzero(dist_mat, axis=1)
    sufficient_controls_mask = valid_controls_per_treated >= n_controls

    if np.any(~sufficient_controls_mask):
        dist_mat = dist_mat[sufficient_controls_mask].reshape(-1, dist_mat.shape[1])
        treated_df = treated_df[sufficient_controls_mask]
    return dist_mat, treated_df
