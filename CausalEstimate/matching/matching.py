import numpy as np
import pandas as pd

from CausalEstimate.matching.distance import (
    compute_distance_matrix,
    filter_treated_w_insufficient_controls,
)
from CausalEstimate.utils.checks import check_required_columns, check_unique_pid
from CausalEstimate.filter.filter import filter_by_column
from CausalEstimate.matching.assignment import (
    assign_controls,
    validate_control_availability,
)
from CausalEstimate.matching.helpers import check_ps_validity


def match_optimal(
    df: pd.DataFrame,
    n_controls: int = 1,
    caliper: float = 0.05,
    treatment_col: str = "treatment",
    ps_col: str = "ps",
    pid_col: str = "PID",
) -> pd.DataFrame:
    """
    Matches treated individuals to control individuals based on propensity scores
    with the option to specify the number of controls per treated individual and a caliper.

    Args:
        df (pd.DataFrame): DataFrame containing treated and control individuals.
        n_controls (int): Number of controls to match for each treated individual.
        caliper (float): Maximum allowable distance (propensity score difference) for matching.
        treatment_col (str): Column name indicating treatment status.
        ps_col (str): Column name for propensity score.
        pid_col (str): Column name for individual ID.

    Returns:
        pd.DataFrame: DataFrame with treated_pid, control_pid and distance columns.
    """
    check_required_columns(df, [treatment_col, ps_col, pid_col])
    check_unique_pid(df, pid_col)
    check_ps_validity(df, ps_col)

    treated_df = filter_by_column(df, treatment_col, 1)
    control_df = filter_by_column(df, treatment_col, 0)

    distance_matrix = compute_distance_matrix(treated_df, control_df, ps_col)

    if caliper is not None:
        distance_matrix[distance_matrix > caliper] = (
            0  # this will ignore all distances greater than the caliper
        )

    distance_matrix, treated_df = filter_treated_w_insufficient_controls(
        distance_matrix, treated_df, n_controls
    )
    validate_control_availability(treated_df, control_df, n_controls)
    # print(dist_mat)
    distance_matrix = np.repeat(
        distance_matrix, repeats=n_controls, axis=0
    )  # repeat the matrix n_controls times
    row_ind, col_ind = assign_controls(distance_matrix)

    matched_distances = distance_matrix[row_ind, col_ind].reshape(
        -1, n_controls
    )  # n_cases x n_controls
    col_ind = col_ind.reshape(-1, n_controls)  # n_cases x n_controls

    result = create_matched_df(
        matched_distances, treated_df, control_df, pid_col, n_controls, col_ind
    )
    return result


def create_matched_df(
    matched_distances: np.array,
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    pid_col: str,
    n_controls: int,
    col_ind: np.array,
) -> pd.DataFrame:
    """
    Creates a DataFrame of matched treated-control pairs and their distances.
    """
    treated_ids_repeated = np.repeat(treated_df[pid_col].values, n_controls)
    control_ids = control_df.iloc[col_ind.flatten()][pid_col].values
    return pd.DataFrame(
        {
            "treated_pid": treated_ids_repeated,
            "control_pid": control_ids,
            "distance": matched_distances.flatten(),
        }
    )
