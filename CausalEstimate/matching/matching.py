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
from CausalEstimate.utils.constants import (
    TREATMENT_COL,
    PS_COL,
    PID_COL,
    TREATED_PID_COL,
    CONTROL_PID_COL,
    DISTANCE_COL,
)


def match_optimal(
    df: pd.DataFrame,
    n_controls: int = 1,
    caliper: float = 0.05,
    treatment_col: str = TREATMENT_COL,
    ps_col: str = PS_COL,
    pid_col: str = PID_COL,
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


def match_eager(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT_COL,
    ps_col: str = PS_COL,
    pid_col: str = PID_COL,
    caliper: float = None,
) -> pd.DataFrame:
    """
    Performs a greedy nearest-neighbor matching based on propensity scores.

    Args:
        df (pd.DataFrame): Input dataframe.
        treatment_col (str): Name of treatment column (1=treated, 0=control).
        ps_col (str): Name of propensity score column.
        pid_col (str): Name of patient ID column.
        caliper (float, optional): Maximum allowed absolute difference in PS for matching.
            If no control is within the caliper, that treated individual remains unmatched.

    Returns:
        pd.DataFrame with columns:
            treated_pid, control_pid, distance
    """
    # Split into treated vs. control
    treated = df.loc[df[treatment_col] == 1, [pid_col, ps_col]].values
    control = df.loc[df[treatment_col] == 0, [pid_col, ps_col]].values

    matches = []
    used_control = set()  # keep track of which controls we've already paired

    for t_pid, t_ps in treated:
        # compute absolute difference for all controls
        ps_diffs = np.abs(control[:, 1] - t_ps)

        # apply caliper if specified
        if caliper is not None:
            valid_mask = ps_diffs <= caliper
            if not valid_mask.any():
                # no control within caliper -> skip this treated subject
                continue
            ps_diffs = ps_diffs[valid_mask]
            valid_control = control[valid_mask]
        else:
            valid_control = control

        if len(valid_control) == 0:
            # no possible match
            continue

        # sort by smallest distance first
        sorted_indices = np.argsort(ps_diffs)
        found_match = False

        for idx in sorted_indices:
            c_pid = valid_control[idx, 0]
            c_ps = valid_control[idx, 1]
            if c_pid not in used_control:
                dist = abs(t_ps - c_ps)
                matches.append([t_pid, c_pid, dist])
                used_control.add(c_pid)
                found_match = True
                break

        # if found_match is False, it means all valid controls in range were used

    return pd.DataFrame(
        matches, columns=[TREATED_PID_COL, CONTROL_PID_COL, DISTANCE_COL]
    )


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
            TREATED_PID_COL: treated_ids_repeated,
            CONTROL_PID_COL: control_ids,
            DISTANCE_COL: matched_distances.flatten(),
        }
    )
