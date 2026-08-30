import numpy as np
import pandas as pd

from CausalEstimate.utils.checks import (
    check_binary_array,
    check_probability_array,
    check_required_columns,
)


def check_ps_not_exact_zero_one(ps: np.ndarray) -> None:
    """
    Reject propensity scores of exactly 0 or 1: their IPW weights are undefined
    and would only reflect the numerical stabilizer.
    """
    if np.any((ps == 0) | (ps == 1)):
        raise ValueError(
            "Propensity scores of exactly 0 or 1 produce undefined IPW weights; "
            "trim them or refit the propensity model first."
        )


def validate_ps_and_treatment(
    df: pd.DataFrame, ps_col: str, treatment_col: str
) -> tuple[int, int]:
    """
    Shared input validation for diagnostics: columns present, treatment binary,
    propensity scores numeric in [0, 1] (NaN rejected), both arms non-empty.

    Returns (n_treated, n_control) so callers share one arm-count definition.
    """
    check_required_columns(df, [ps_col, treatment_col])
    check_binary_array(df[treatment_col].to_numpy(), "Treatment")
    check_probability_array(df[ps_col].to_numpy(), "Propensity Score")
    n_treated = int((df[treatment_col] == 1).sum())
    n_control = int((df[treatment_col] == 0).sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            "Both treated and control groups must be non-empty "
            f"(n_treated={n_treated}, n_control={n_control})."
        )
    return n_treated, n_control
