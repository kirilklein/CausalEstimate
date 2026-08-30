from typing import List, Literal, Optional

import pandas as pd

from CausalEstimate.diagnostics.balance import check_balance, compute_balance_table
from CausalEstimate.diagnostics.positivity import compute_positivity_metrics
from CausalEstimate.diagnostics.weights import compute_weight_diagnostics
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL


def run_diagnostics(
    df: pd.DataFrame,
    ps_col: str = PS_COL,
    treatment_col: str = TREATMENT_COL,
    covariate_cols: Optional[List[str]] = None,
    weight_type: Literal["ATE", "ATT"] = "ATE",
    clip_percentile: float = 1,
    eps: float = 0.01,
    common_support_threshold: float = 0.05,
    smd_threshold: float = 0.1,
) -> dict:
    """
    Run all propensity-score diagnostics in one call.

    Composes compute_positivity_metrics, compute_weight_diagnostics and — when
    covariate_cols is given — compute_balance_table + check_balance. See each
    function for its metrics and conventions.

    Args:
        df: Input DataFrame with treatment and propensity score columns.
        ps_col: Name of the propensity score column.
        treatment_col: Name of the treatment status column (1 treated, 0 control).
        covariate_cols: Covariates for the balance table; None skips balance.
        weight_type: "ATE" or "ATT", passed to the weight and balance diagnostics.
        clip_percentile: Upper-tail weight clipping passed through.
        eps: Extreme-PS bound for positivity metrics.
        common_support_threshold: Quantile threshold for the trimmed common support.
        smd_threshold: |SMD| bound for the balance table's balanced column.

    Returns:
        {"positivity": dict, "weights": dict, "balance": DataFrame | None,
         "balance_summary": dict | None,
         "flags": {"extreme_ps": bool, "unbalanced": bool | None}} —
        flags["unbalanced"] is None when balance was skipped.
    """
    positivity = compute_positivity_metrics(
        df,
        ps_col=ps_col,
        treatment_col=treatment_col,
        eps=eps,
        common_support_threshold=common_support_threshold,
    )
    weights = compute_weight_diagnostics(
        df,
        ps_col=ps_col,
        treatment_col=treatment_col,
        weight_type=weight_type,
        clip_percentile=clip_percentile,
    )
    balance = None
    balance_summary = None
    if covariate_cols is not None:
        balance = compute_balance_table(
            df,
            covariate_cols,
            ps_col=ps_col,
            treatment_col=treatment_col,
            weight_type=weight_type,
            clip_percentile=clip_percentile,
            threshold=smd_threshold,
        )
        balance_summary = check_balance(balance)
    return {
        "positivity": positivity,
        "weights": weights,
        "balance": balance,
        "balance_summary": balance_summary,
        "flags": {
            "extreme_ps": positivity["flag_extreme_ps"],
            "unbalanced": (
                None if balance_summary is None else not balance_summary["balanced"]
            ),
        },
    }
