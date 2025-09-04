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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, clip_percentile: float = 1
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes(A, Y, ps, clip_percentile=clip_percentile)
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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, clip_percentile: float = 1
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes(A, Y, ps, clip_percentile=clip_percentile)
    ate = mu_1 - mu_0
    return {EFFECT: ate, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_att(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, clip_percentile: float = 1
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes_treated(
        A, Y, ps, clip_percentile=clip_percentile
    )
    att = mu_1 - mu_0
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_risk_ratio_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, clip_percentile: float = 1
) -> dict:
    """
    Computes the Relative Risk for the Treated (RRT) using IPW.
    """
    mu_1, mu_0 = compute_weighted_outcomes_treated(
        A, Y, ps, clip_percentile=clip_percentile
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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, clip_percentile: float = 1
) -> Tuple[float, float]:
    """
    Computes E[Y(1)] and E[Y(0)] for the ATE using the simple Horvitz-Thompson estimator,
    with explicit checks for empty groups.
    """
    W = compute_ipw_weights(A, ps, weight_type="ATE", clip_percentile=clip_percentile)

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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, clip_percentile: float = 1
) -> Tuple[float, float]:
    """
    Computes E[Y(1)|A=1] and E[Y(0)|A=1] for the ATT using the robust Hajek (ratio) estimator.
    """
    W = compute_ipw_weights(A, ps, weight_type="ATT", clip_percentile=clip_percentile)

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
) -> np.ndarray:
    """
    Compute IPW weights for ATE or ATT with optional stabilization for ATE.
    """

    if weight_type == "ATE":
        weight_treated = 1 / ps
        weight_control = 1 / (1 - ps)
        if clip_percentile < 1:
            treated_mask = A == 1
            control_mask = A == 0
            threshold = np.percentile(
                weight_treated[treated_mask], clip_percentile * 100
            )  # only compute threshold for treated group
            weight_treated = np.clip(weight_treated, a_min=None, a_max=threshold)
            threshold = np.percentile(
                weight_control[control_mask], clip_percentile * 100
            )  # only compute threshold for control group
            weight_control = np.clip(weight_control, a_min=None, a_max=threshold)
        weights = np.where(A == 1, weight_treated, weight_control)

    elif weight_type == "ATT":
        weight_treated = np.ones_like(A, dtype=float)
        weight_control = ps / (1 - ps)
        weights = np.where(A == 1, weight_treated, weight_control)
        if clip_percentile < 1:
            # Only clip the weights for the control group
            control_weights = weights[A == 0]
            if control_weights.size > 0:
                threshold = np.percentile(control_weights, clip_percentile * 100)
                weights[A == 0] = np.clip(control_weights, a_min=None, a_max=threshold)
    else:
        raise ValueError("weight_type must be 'ATE' or 'ATT'")
    return weights
