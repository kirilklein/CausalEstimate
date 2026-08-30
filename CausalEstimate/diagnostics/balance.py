import warnings
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from CausalEstimate.diagnostics.utils import validate_ps_and_treatment
from CausalEstimate.estimators.functional.ipw import compute_ipw_weights
from CausalEstimate.utils.checks import (
    check_binary_array,
    check_columns_for_nans,
    check_required_columns,
)
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


def _check_threshold(threshold: float) -> None:
    if not (np.isfinite(threshold) and threshold > 0):
        raise ValueError(
            f"threshold must be a finite positive number, got {threshold}."
        )


def compute_smd(
    x: np.ndarray, A: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Standardized mean difference of a covariate between treated and control.

    Convention: the denominator is the pooled *unweighted* sample SD,
    sqrt((s1^2 + s0^2)/2), held fixed for both the unweighted and the weighted
    SMD so pre- and post-weighting values share a scale — the "standardize by
    the unadjusted sample" convention recommended by Stuart (2010) and used as
    the default in the cobalt R package. Note this deliberately differs from
    Austin & Stuart (2015), who use weighted variances post-weighting and a
    prevalence-based formula for binary covariates; binary covariates here go
    through the same numeric formula (sample variance ~ p(1-p)). A
    zero-variance covariate has no defined SMD: a warning is emitted and NaN
    returned.

    Parameters:
    -----------
    x : Covariate values (finite).
    A : Binary treatment assignment (1 treated, 0 control); each arm needs at
        least 2 observations.
    weights : Optional finite, nonnegative weights with positive sum per arm;
              when given, the mean difference is weighted (the denominator
              stays unweighted).
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A)
    if x.shape != A.shape:
        raise ValueError("x and A must have the same shape.")
    check_binary_array(A, "Treatment")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite (no NaN or inf).")
    x_treated, x_control = x[A == 1], x[A == 0]
    if len(x_treated) < 2 or len(x_control) < 2:
        raise ValueError(
            "Each arm needs at least 2 observations to compute an SMD "
            f"(n_treated={len(x_treated)}, n_control={len(x_control)})."
        )
    if weights is None:
        diff = x_treated.mean() - x_control.mean()
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != A.shape:
            raise ValueError("weights and A must have the same shape.")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and nonnegative.")
        w_treated, w_control = w[A == 1], w[A == 0]
        if w_treated.sum() == 0 or w_control.sum() == 0:
            raise ValueError("weights must have a positive sum in each arm.")
        diff = np.average(x_treated, weights=w_treated) - np.average(
            x_control, weights=w_control
        )
    pooled_var = (x_treated.var(ddof=1) + x_control.var(ddof=1)) / 2
    if pooled_var == 0:
        warnings.warn(
            "Zero-variance covariate: SMD is undefined, returning NaN.",
            RuntimeWarning,
        )
        return float("nan")
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
    (|smd_weighted| < threshold; False where the SMD is NaN, i.e. undefined is
    never certified as balanced). Weights come from compute_ipw_weights with
    the given weight_type and clip_percentile; see compute_smd for the SMD
    convention. For weight_type="ATT" the denominator is still the pooled
    unweighted SD of both arms — the treated-arm-SD variant is deliberately
    not implemented yet.
    """
    if not covariate_cols:
        raise ValueError("covariate_cols must not be empty.")
    _check_threshold(threshold)
    validate_ps_and_treatment(df, ps_col, treatment_col)
    check_required_columns(df, covariate_cols)
    check_columns_for_nans(df, covariate_cols)
    A = df[treatment_col].to_numpy()
    ps = df[ps_col].to_numpy()

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


def check_balance(balance_table: pd.DataFrame) -> dict:
    """
    Summarize a compute_balance_table result.

    The unbalanced/balanced classification is read from the table's balanced
    column, so the summary always agrees with the threshold the table was
    built with. NaN SMDs (zero-variance covariates) are counted in n_undefined
    and excluded from max_smd_weighted and prop_unbalanced (which is over
    evaluable covariates only, NaN when none are evaluable). balanced is True
    only when no covariate is unbalanced AND none is undefined.
    """
    if len(balance_table) == 0:
        raise ValueError("balance_table must not be empty.")
    abs_smd = balance_table[SMD_WEIGHTED_COL].abs()
    n_undefined = int(abs_smd.isna().sum())
    evaluable = abs_smd.dropna()
    n_unbalanced = int((~balance_table[BALANCED_COL] & abs_smd.notna()).sum())
    return {
        "max_smd_weighted": (
            float(evaluable.max()) if len(evaluable) else float("nan")
        ),
        "n_unbalanced": n_unbalanced,
        "n_undefined": n_undefined,
        "prop_unbalanced": (
            n_unbalanced / len(evaluable) if len(evaluable) else float("nan")
        ),
        "balanced": bool(n_unbalanced == 0 and n_undefined == 0),
    }
