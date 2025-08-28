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
    treatment_col: str = TREATMENT_COL,
    ps_col: str = PS_COL,
    pid_col: str = PID_COL,
    n_controls: int = 1,
    caliper: float = None,
) -> pd.DataFrame:
    """
    Matches treated individuals to control individuals based on propensity scores
    with the option to specify the number of controls per treated individual and a caliper.

    This function uses optimal matching to minimize the total distance between treated
    and control subjects, which typically produces better overall balance than greedy
    matching approaches.

    Args:
        df (pd.DataFrame): DataFrame containing treated and control individuals.
        treatment_col (str): Column name indicating treatment status.
        ps_col (str): Column name for propensity score.
        pid_col (str): Column name for individual ID.
        n_controls (int): Number of controls to match for each treated individual.
                         Must be >= 1. Common values:
                         - 1: 1:1 matching (most common, maximizes precision)
                         - 2-5: small ratios for bias-variance tradeoff
                         - 10+: large ratios when controls are abundant
        caliper (float): Maximum allowable distance (propensity score difference) for matching.
                        Must be >= 0 when provided. If None, no caliper is applied.
                        Common values:
                        - 0.1: loose caliper, allows moderate PS differences
                        - 0.05: moderate caliper, good balance of matches vs quality
                        - 0.01-0.02: tight caliper, ensures close PS matches
                        - 0.25*std(PS): standard recommendation (Rosenbaum & Rubin, 1985)

    Returns:
        pd.DataFrame: DataFrame with treated_pid, control_pid and distance columns.

    Raises:
        ValueError: If n_controls < 1 or caliper < 0.
    """
    check_required_columns(df, [treatment_col, ps_col, pid_col])
    check_unique_pid(df, pid_col)
    check_ps_validity(df, ps_col)

    # Parameter validation
    if n_controls < 1:
        raise ValueError("n_controls must be >= 1")
    if caliper is not None and caliper < 0:
        raise ValueError("caliper must be >= 0 when provided")

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
    n_controls: int = 1,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Performs a greedy nearest-neighbor matching based on propensity scores,
    allowing multiple controls per treated subject. By default, n_controls=1.

    This function uses a greedy approach that matches each treated subject to their
    nearest available control(s) in order. While faster than optimal matching,
    it may not achieve the best overall balance across all matches.

    Matching proceeds in multiple "passes":
    - Pass 1: each treated tries to find its first best control
    - Pass 2: each treated tries to find its second best control, etc.

    Args:
        df (pd.DataFrame): Input dataframe.
        treatment_col (str): Name of treatment column (1=treated, 0=control).
        ps_col (str): Name of propensity score column.
        pid_col (str): Name of patient ID column.
        caliper (float, optional): Maximum allowed absolute difference in PS for matching.
                                   Must be >= 0 when provided. If no control is within the caliper,
                                   that treated subject remains unmatched (or raises ValueError if strict=True).
                                   Common values:
                                   - 0.1: loose caliper, allows moderate PS differences
                                   - 0.05: moderate caliper, good balance of matches vs quality
                                   - 0.01-0.02: tight caliper, ensures close PS matches
        n_controls (int): How many distinct control matches to find per treated subject.
                         Must be >= 1. Common values:
                         - 1: 1:1 matching (most common, maximizes precision)
                         - 2-5: small ratios for bias-variance tradeoff
                         - 10+: large ratios when controls are abundant
        strict (bool): If True, raise a ValueError if any treated subject fails to find
                       a control at any pass. If False, skip unmatched passes.

    Returns:
        pd.DataFrame with columns [treated_pid, control_pid, distance].
        This may contain up to (n_controls * number_of_treated) rows,
        if all can be matched on every pass.

    Raises:
        ValueError: If strict=True and a treated subject cannot be matched on any pass,
                   or if n_controls < 1 or caliper < 0.
    """
    # Parameter validation
    if n_controls < 1:
        raise ValueError("n_controls must be >= 1")
    if caliper is not None and caliper < 0:
        raise ValueError("caliper must be >= 0 when provided")

    # Separate treated vs. control
    treated_array = df.loc[df[treatment_col] == 1, [pid_col, ps_col]].values
    control_array = df.loc[df[treatment_col] == 0, [pid_col, ps_col]].values

    # Keep track of how many matches each treated has found
    # Map from treated_pid -> current match count
    # (only needed if we want to track partial matches in strict mode)
    match_count = {t_pid: 0 for t_pid, _ in treated_array}

    all_matches = []
    used_control = set()  # keep track of which controls are already matched

    # Repeat the greedy pass n_controls times
    for pass_index in range(n_controls):
        pass_matches = []  # store matches found in this pass

        for t_pid, t_ps in treated_array:
            # If this treated subject is already fully matched in previous passes, skip
            if match_count[t_pid] >= pass_index + 1:
                # already got pass_index matches
                continue

            # compute distance to all controls
            ps_diffs = np.abs(control_array[:, 1] - t_ps)

            # apply caliper if specified
            if caliper is not None:
                within_caliper = ps_diffs <= caliper
                if not within_caliper.any():
                    if strict:
                        raise ValueError(
                            f"Treated subject {t_pid} cannot find a control within caliper={caliper} "
                            f"on pass {pass_index+1} of {n_controls}."
                        )
                    else:
                        continue
                ps_diffs = ps_diffs[within_caliper]
                valid_control = control_array[within_caliper]
            else:
                valid_control = control_array

            if len(valid_control) == 0:
                if strict:
                    raise ValueError(
                        f"Treated subject {t_pid} cannot find any available control on pass {pass_index+1}. "
                        f"All possible controls are used or out of range."
                    )
                else:
                    continue

            # sort by smallest distance
            sorted_indices = np.argsort(ps_diffs)
            found_match = False

            for idx in sorted_indices:
                c_pid = valid_control[idx, 0]
                c_ps = valid_control[idx, 1]
                if c_pid not in used_control:
                    dist = abs(t_ps - c_ps)
                    pass_matches.append([t_pid, c_pid, dist])
                    used_control.add(c_pid)  # can't use it again
                    match_count[t_pid] += 1
                    found_match = True
                    break

            if not found_match and strict:
                raise ValueError(
                    f"Treated subject {t_pid} cannot find an available control on pass {pass_index+1}, "
                    "either because all valid controls are used up or out of range."
                )

        # merge pass_matches into our global list
        all_matches.extend(pass_matches)

    return pd.DataFrame(
        all_matches, columns=[TREATED_PID_COL, CONTROL_PID_COL, DISTANCE_COL]
    )


def create_matched_df(
    matched_distances: np.ndarray,
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    pid_col: str,
    n_controls: int,
    col_ind: np.ndarray,
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
