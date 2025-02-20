import pandas as pd
from CausalEstimate.utils.utils import get_treated_ps, get_untreated_ps, filter_column
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL


def filter_common_support(
    df: pd.DataFrame,
    ps_col: str = PS_COL,
    treatment_col: str = TREATMENT_COL,
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
    common_min, common_max = get_common_support_range(
        df, treatment_col, ps_col, threshold
    )
    filtered_df = filter_column(df, ps_col, common_min, common_max)
    return filtered_df


def get_common_support_range(
    df: pd.DataFrame, treatment_col: str, ps_col: str, threshold: float = 0.05
) -> tuple[float, float]:
    """
    Calculate the common support range for propensity scores.

    Parameters:
    -----------
    df : Input DataFrame with treatment and propensity score columns.
    treatment_col : Name of the treatment status column.
    ps_col : Name of the propensity score column.
    threshold : Quantile threshold for trimming score distribution tails. Default is 0.05.

    Returns:
    --------
    Lower and upper bounds of the common support range.
    """
    min_ps_treated, max_ps_treated = get_treated_ps(df, treatment_col, ps_col).quantile(
        [threshold, 1 - threshold]
    )
    min_ps_control, max_ps_control = get_untreated_ps(
        df, treatment_col, ps_col
    ).quantile([threshold, 1 - threshold])
    common_min = max(min_ps_treated, min_ps_control)
    common_max = min(max_ps_treated, max_ps_control)
    return common_min, common_max
