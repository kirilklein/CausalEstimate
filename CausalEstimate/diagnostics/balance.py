import warnings
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from CausalEstimate.diagnostics.utils import (
    check_ps_not_exact_zero_one,
    validate_ps_and_treatment,
)
from CausalEstimate.estimators.functional.ipw import compute_ipw_weights
from CausalEstimate.utils.checks import check_columns_for_nans, check_required_columns
from CausalEstimate.utils.constants import (
    BALANCED_COL,
    COVARIATE_COL,
    MEAN_CONTROL_COL,
    MEAN_TREATED_COL,
    PS_COL,
    SMD_UNWEIGHTED_COL,
    SMD_WEIGHTED_COL,
    TREATMENT_COL,
    WEIGHTED_MEAN_CONTROL_COL,
    WEIGHTED_MEAN_TREATED_COL,
)


def compute_smd(
    x: np.ndarray, A: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Standardized mean difference of a covariate between treated and control.

    The denominator is the pooled *unweighted* SD, sqrt((s1^2 + s0^2)/2), for
    both the unweighted and the weighted SMD, so pre- and post-weighting values
    share a scale (Austin & Stuart 2015). Binary covariates go through the same
    numeric formula. A zero-variance covariate has no defined SMD: a warning is
    emitted and NaN returned.

    Parameters:
    -----------
    x : Covariate values.
    A : Binary treatment assignment (1 treated, 0 control).
    weights : Optional weights; when given, the mean difference is weighted
              (the denominator stays unweighted).
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A)
    x_treated, x_control = x[A == 1], x[A == 0]
    pooled_var = (x_treated.var(ddof=1) + x_control.var(ddof=1)) / 2
    if pooled_var == 0:
        warnings.warn(
            "Zero-variance covariate: SMD is undefined, returning NaN.",
            RuntimeWarning,
        )
        return float("nan")
    if weights is None:
        diff = x_treated.mean() - x_control.mean()
    else:
        w = np.asarray(weights, dtype=float)
        diff = np.average(x_treated, weights=w[A == 1]) - np.average(
            x_control, weights=w[A == 0]
        )
    return float(diff / np.sqrt(pooled_var))


def compute_balance_table(
    df: pd.DataFrame,
    covariate_cols: List[str],
    ps_col: str = PS_COL,
    treatment_col: str = TREATMENT_COL,
    weight_type: Literal["ATE", "ATT"] = "ATE",
    clip_percentile: float = 1,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Covariate balance table before and after IPW weighting.

    One row per covariate, indexed by covariate name: unweighted and weighted
    means per arm, smd_unweighted, smd_weighted, and balanced
    (|smd_weighted| < threshold; False where the SMD is NaN). Weights come from
    compute_ipw_weights with the given weight_type and clip_percentile; see
    compute_smd for the SMD convention.
    """
    validate_ps_and_treatment(df, ps_col, treatment_col)
    check_required_columns(df, covariate_cols)
    check_columns_for_nans(df, covariate_cols)
    A = df[treatment_col].to_numpy()
    ps = df[ps_col].to_numpy()
    check_ps_not_exact_zero_one(ps)

    W = compute_ipw_weights(
        A, ps, weight_type=weight_type, clip_percentile=clip_percentile
    )
    w_treated, w_control = W[A == 1], W[A == 0]

    rows = []
    for col in covariate_cols:
        x = df[col].to_numpy(dtype=float)
        x_treated, x_control = x[A == 1], x[A == 0]
        smd_weighted = compute_smd(x, A, W)
        rows.append(
            {
                COVARIATE_COL: col,
                MEAN_TREATED_COL: float(x_treated.mean()),
                MEAN_CONTROL_COL: float(x_control.mean()),
                WEIGHTED_MEAN_TREATED_COL: float(
                    np.average(x_treated, weights=w_treated)
                ),
                WEIGHTED_MEAN_CONTROL_COL: float(
                    np.average(x_control, weights=w_control)
                ),
                SMD_UNWEIGHTED_COL: compute_smd(x, A),
                SMD_WEIGHTED_COL: smd_weighted,
                BALANCED_COL: bool(abs(smd_weighted) < threshold),
            }
        )
    return pd.DataFrame(rows).set_index(COVARIATE_COL)


def check_balance(balance_table: pd.DataFrame, threshold: float = 0.1) -> dict:
    """
    Summarize a compute_balance_table result.

    NaN SMDs (zero-variance covariates) are excluded from the maximum and not
    counted as unbalanced.
    """
    abs_smd = balance_table[SMD_WEIGHTED_COL].abs()
    n_unbalanced = int((abs_smd >= threshold).sum())
    return {
        "max_smd_weighted": float(abs_smd.max()),
        "n_unbalanced": n_unbalanced,
        "prop_unbalanced": n_unbalanced / len(balance_table),
        "balanced": bool(n_unbalanced == 0),
    }
