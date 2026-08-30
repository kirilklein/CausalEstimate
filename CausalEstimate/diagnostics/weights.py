from typing import Literal

import numpy as np
import pandas as pd

from CausalEstimate.diagnostics.utils import validate_ps_and_treatment
from CausalEstimate.estimators.functional.ipw import compute_ipw_weights
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL


def compute_ess(weights: np.ndarray) -> float:
    """
    Effective sample size of a weighted sample (Kish): (sum w)^2 / sum(w^2).

    Equals n for uniform weights and approaches 1 as a single weight dominates.
    Defined for nonnegative weights; nonfinite, negative, or all-zero weights
    raise ValueError.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        raise ValueError("weights must be non-empty.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite (no NaN or inf).")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative.")
    if w.max() == 0:
        raise ValueError("weights must not be all zero.")
    w = w / w.max()  # ESS is scale-invariant; normalizing prevents overflow
    return float(w.sum() ** 2 / (w**2).sum())


def _weight_summary(w: np.ndarray, suffix: str) -> dict:
    return {
        f"max_weight_{suffix}": float(w.max()),
        f"mean_weight_{suffix}": float(w.mean()),
        f"weight_q95_{suffix}": float(np.percentile(w, 95)),
        f"weight_q99_{suffix}": float(np.percentile(w, 99)),
    }


def compute_weight_diagnostics(
    df: pd.DataFrame,
    ps_col: str = PS_COL,
    treatment_col: str = TREATMENT_COL,
    weight_type: Literal["ATE", "ATT"] = "ATE",
    clip_percentile: float = 1,
) -> dict:
    """
    Compute IPW weight diagnostics: effective sample size and weight summaries.

    Weights are computed with compute_ipw_weights; clip_percentile=1 (default)
    means raw, unclipped weights, matching the estimators' default.
    Propensity scores of exactly 0 or 1 raise ValueError: their IPW weights are
    undefined and would only reflect the numerical stabilizer.

    Args:
        df: Input DataFrame with treatment and propensity score columns.
        ps_col: Name of the propensity score column.
        treatment_col: Name of the treatment status column (1 treated, 0 control).
        weight_type: "ATE" or "ATT".
        clip_percentile: Upper-tail clipping passed to compute_ipw_weights.

    Returns:
        dict with per-arm ESS, ESS as a fraction of arm size, and per-arm weight
        summaries (max, mean, 95th and 99th percentile). Pooled equivalents
        (ess_total, ess_fraction_total, max_weight, mean_weight, weight_q95,
        weight_q99) are included only for weight_type="ATE": ATT weights put the
        two arms on different scales (treated weights are identically 1), so
        pooled Kish ESS and pooled quantiles would be misleading there.
    """
    n_treated, n_control = validate_ps_and_treatment(df, ps_col, treatment_col)
    A = df[treatment_col].to_numpy()
    ps = df[ps_col].to_numpy()
    if np.any((ps == 0) | (ps == 1)):
        raise ValueError(
            "Propensity scores of exactly 0 or 1 produce undefined IPW weights; "
            "trim them or refit the propensity model before weight diagnostics."
        )

    W = compute_ipw_weights(
        A, ps, weight_type=weight_type, clip_percentile=clip_percentile
    )
    W_treated = W[A == 1]
    W_control = W[A == 0]
    ess_treated = compute_ess(W_treated)
    ess_control = compute_ess(W_control)

    result = {
        "n_total": int(len(W)),
        "n_treated": n_treated,
        "n_control": n_control,
        "ess_treated": ess_treated,
        "ess_control": ess_control,
        "ess_fraction_treated": min(ess_treated / n_treated, 1.0),
        "ess_fraction_control": min(ess_control / n_control, 1.0),
        **_weight_summary(W_treated, "treated"),
        **_weight_summary(W_control, "control"),
    }
    if weight_type == "ATE":
        ess_total = compute_ess(W)
        result.update(
            {
                "ess_total": ess_total,
                "ess_fraction_total": min(ess_total / len(W), 1.0),
                "max_weight": float(W.max()),
                "mean_weight": float(W.mean()),
                "weight_q95": float(np.percentile(W, 95)),
                "weight_q99": float(np.percentile(W, 99)),
            }
        )
    return result
