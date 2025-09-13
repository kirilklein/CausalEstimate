"""
Inverse Probability Weighting (IPW) estimators

References:
ATE:
    Estimation of Average Treatment Effects Honors Thesis Peter Zhang
    https://lsa.umich.edu/content/dam/econ-assets/Econdocs/HonorsTheses/Estimation%20of%20Average%20Treatment%20Effects.pdf

    Austin, P.C., 2016. Variance estimation when using inverse probability of
    treatment weighting (IPTW) with survival analysis.
    Statistics in medicine, 35(30), pp.5642-5655.

ATT:
    Reifeis et. al. (2022).
    On variance of the treatment effect in the treated when estimated by
    inverse probability weighting.
    American Journal of Epidemiology, 191(6), 1092-1097.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9271225/

We also provide an option to use stabilized weights as described in:
Miguel A Hernán 1, James M Robins
Estimating causal effects from epidemiological data
https://pubmed.ncbi.nlm.nih.gov/16790829/
"""

import warnings
from typing import Tuple, Literal

import numpy as np

from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated

# --- Core Effect Calculation Functions ---


def compute_ipw_risk_ratio(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes(
        A, Y, ps, clip_percentile=clip_percentile, eps=eps
    )
    if mu_0 == 0:
        warnings.warn(
            "Risk in untreated group (mu_0) is 0, returning inf for Risk Ratio.",
            RuntimeWarning,
        )
        rr = np.inf
    else:
        rr = mu_1 / mu_0
    return {EFFECT: rr, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_ate(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes(
        A, Y, ps, clip_percentile=clip_percentile, eps=eps
    )
    ate = mu_1 - mu_0
    return {EFFECT: ate, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_att(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes_treated(
        A, Y, ps, clip_percentile=clip_percentile, eps=eps
    )
    att = mu_1 - mu_0
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_risk_ratio_treated(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Computes the Relative Risk for the Treated (RRT) using IPW.
    """
    mu_1, mu_0 = compute_weighted_outcomes_treated(
        A, Y, ps, clip_percentile=clip_percentile, eps=eps
    )
    if mu_0 == 0:
        warnings.warn(
            "Risk in counterfactual untreated group (mu_0) is 0, returning inf for RRT.",
            RuntimeWarning,
        )
        rrt = np.inf
    else:
        rrt = mu_1 / mu_0
    return {EFFECT: rrt, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


# --- Weighted Mean Estimators (Refactored) ---


def compute_weighted_outcomes(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> Tuple[float, float]:
    """
    Computes E[Y(1)] and E[Y(0)] for the ATE using the simple Horvitz-Thompson estimator,
    with explicit checks for empty groups.
    """
    W = compute_ipw_weights(
        A, ps, weight_type="ATE", clip_percentile=clip_percentile, eps=eps
    )

    # --- Calculate for Treated Group (mu_1) ---
    treated_mask: np.ndarray = A == 1

    if treated_mask.sum() > 0:
        numerator_1 = (W[treated_mask] * Y[treated_mask]).sum()
        denominator_1 = W[treated_mask].sum()
        mu_1 = numerator_1 / denominator_1 if denominator_1 != 0 else np.nan
    else:
        warnings.warn("No subjects in the treated group. mu_1 is NaN.", RuntimeWarning)
        mu_1 = np.nan

    # --- Calculate for Control Group (mu_0) ---
    control_mask: np.ndarray = A == 0
    if control_mask.sum() > 0:
        numerator_0 = (W[control_mask] * Y[control_mask]).sum()
        denominator_0 = W[control_mask].sum()
        mu_0 = numerator_0 / denominator_0 if denominator_0 != 0 else np.nan
    else:
        warnings.warn("No subjects in the control group. mu_0 is NaN.", RuntimeWarning)
        mu_0 = np.nan

    return mu_1, mu_0


def compute_weighted_outcomes_treated(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> Tuple[float, float]:
    """
    Computes E[Y(1)|A=1] and E[Y(0)|A=1] for the ATT using the robust Hajek (ratio) estimator.
    """
    W = compute_ipw_weights(
        A, ps, weight_type="ATT", clip_percentile=clip_percentile, eps=eps
    )

    # --- Factual Outcome for the Treated (mu_1) ---
    treated_mask: np.ndarray = A == 1
    num_treated = treated_mask.sum()
    if num_treated > 0:
        mu_1 = Y[treated_mask].mean()  # No adjustment for treated
    else:
        warnings.warn(
            "No subjects in the treated group for ATT. mu_1 is NaN.", RuntimeWarning
        )
        mu_1 = np.nan

    # --- Counterfactual Outcome for the Treated (mu_0) ---
    control_mask: np.ndarray = A == 0
    if num_treated > 0 and control_mask.sum() > 0:
        weights_control = W[control_mask]
        outcomes_control = Y[control_mask]

        numerator_0 = (weights_control * outcomes_control).sum()
        denominator_0 = weights_control.sum()

        mu_0 = numerator_0 / denominator_0 if denominator_0 != 0 else np.nan
    else:
        if num_treated == 0:
            warnings.warn(
                "No subjects in the treated group for ATT. mu_0 is NaN.", RuntimeWarning
            )
        else:  # Implies no controls
            warnings.warn(
                "No subjects in the control group for ATT. mu_0 is NaN.", RuntimeWarning
            )
        mu_0 = np.nan

    return mu_1, mu_0


# --- Centralized Weight Calculation Functions ---


def compute_ipw_weights(
    A: np.ndarray,
    ps: np.ndarray,
    weight_type: Literal["ATE", "ATT"] = "ATE",
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Computes Inverse Propensity Score (IPW) weights with optional clipping.

    This function calculates weights for estimating the Average Treatment Effect (ATE)
    or the Average Treatment Effect on the Treated (ATT).

    Formulas:
    - ATE: w = A/ps + (1-A)/(1-ps)
    - ATT: w = A + (1-A) * ps/(1-ps)

        Args:
        A: Binary treatment assignment vector (1 for treated, 0 for control).
        ps: Propensity score vector (estimated probability of treatment).
        weight_type: The type of estimand, either "ATE" or "ATT".
        clip_percentile: The upper percentile at which to clip weights to prevent
                         extreme values. A value of 1.0 (default) applies no
                         clipping. For example, 0.99 clips the top 1%.
        eps: A small constant to add to denominators for numerical stability.

    Returns:
        An array of computed IPW weights.

    Raises:
        ValueError: If `weight_type` is invalid, shapes mismatch, or
                    `clip_percentile` is out of bounds.
    """

    # --- 1. Input Validation ---
    if weight_type not in ["ATE", "ATT"]:
        raise ValueError("weight_type must be 'ATE' or 'ATT'")
    if not (0 < clip_percentile <= 1.0):
        raise ValueError("clip_percentile must be in the interval (0, 1.0].")
    if A.shape != ps.shape:
        raise ValueError("A and ps must have the same shape.")

    # --- 2. Core Weight Calculation ---
    if weight_type == "ATE":
        # Vectorized formula for ATE weights
        weights = A / (ps + eps) + (1 - A) / (1 - ps + eps)
    else:  # weight_type == "ATT"
        # For ATT, treated units have a weight of 1.
        # Vectorized formula for ATT weights.
        weights = A + (1 - A) * ps / (1 - ps + eps)

    # --- 3. Unified Weight Clipping ---
    if clip_percentile < 1.0:
        q = clip_percentile * 100

        # This logic is now applied to both ATE and ATT.
        # For ATT, the 'treated_mask' section is a no-op but is still executed.
        treated_mask = A == 1
        if np.any(treated_mask):
            threshold_t = np.percentile(weights[treated_mask], q)
            weights[treated_mask] = np.clip(
                weights[treated_mask], a_min=None, a_max=threshold_t
            )

        control_mask = ~treated_mask
        if np.any(control_mask):
            threshold_c = np.percentile(weights[control_mask], q)
            weights[control_mask] = np.clip(
                weights[control_mask], a_min=None, a_max=threshold_c
            )

    return weights
