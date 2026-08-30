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
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        raise ValueError("weights must be non-empty.")
    return float(w.sum() ** 2 / (w**2).sum())


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

    Parameters:
    -----------
    df : Input DataFrame with treatment and propensity score columns.
    ps_col : Name of the propensity score column.
    treatment_col : Name of the treatment status column (1 treated, 0 control).
    weight_type : "ATE" or "ATT".
    clip_percentile : Upper-tail clipping passed to compute_ipw_weights.

    Returns:
    --------
    dict with ESS (total and per arm), ESS as a fraction of n, and weight
    summaries (max, mean, 95th and 99th percentile), overall and per arm.
    """
    validate_ps_and_treatment(df, ps_col, treatment_col)
    A = df[treatment_col].to_numpy()
    ps = df[ps_col].to_numpy()
    n_treated = int((A == 1).sum())
    n_control = int((A == 0).sum())

    W = compute_ipw_weights(
        A, ps, weight_type=weight_type, clip_percentile=clip_percentile
    )
    W_treated = W[A == 1]
    W_control = W[A == 0]

    return {
        "n_total": int(len(W)),
        "n_treated": n_treated,
        "n_control": n_control,
        "ess_total": compute_ess(W),
        "ess_treated": compute_ess(W_treated),
        "ess_control": compute_ess(W_control),
        "ess_fraction_total": compute_ess(W) / len(W),
        "ess_fraction_treated": compute_ess(W_treated) / n_treated,
        "ess_fraction_control": compute_ess(W_control) / n_control,
        "max_weight": float(W.max()),
        "mean_weight": float(W.mean()),
        "weight_q95": float(np.percentile(W, 95)),
        "weight_q99": float(np.percentile(W, 99)),
        "max_weight_treated": float(W_treated.max()),
        "max_weight_control": float(W_control.max()),
    }
