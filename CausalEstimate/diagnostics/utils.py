import pandas as pd

from CausalEstimate.utils.checks import (
    check_binary_array,
    check_probability_array,
    check_required_columns,
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
