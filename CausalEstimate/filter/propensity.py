import pandas as pd


def filter_common_support(
    df: pd.DataFrame,
    ps_col: str = "propensity_score",
    treatment_col: str = "treatment",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Filters individuals based on common support in propensity scores, removing those outside the range.

    Parameters:
    df: Input DataFrame containing columns for PID, propensity score, and treatment status.
    pid_col: Column name for the participant ID.
    ps_col: Column name for the propensity score.
    treatment_col: Column name for the treatment status (1 for treated, 0 for control).
    threshold: Optional threshold in quantile (default 0.05) to trim the tails of the distribution for better common support.

    Returns:
    DataFrame after removing individuals without common support.
    """
    # Split the dataframe into treated and control groups
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    # Get the range of propensity scores for treated and control groups
    min_ps_treated, max_ps_treated = treated[ps_col].quantile(
        [threshold, 1 - threshold]
    )
    min_ps_control, max_ps_control = control[ps_col].quantile(
        [threshold, 1 - threshold]
    )
    # Define the common support range
    common_min = max(min_ps_treated, min_ps_control)
    common_max = min(max_ps_treated, max_ps_control)
    # Filter individuals to keep only those within the common support range
    filtered_df = df[(df[ps_col] >= common_min) & (df[ps_col] <= common_max)]
    return filtered_df
