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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes(A, Y, ps, stabilized=stabilized)
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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes(A, Y, ps, stabilized=stabilized)
    ate = mu_1 - mu_0
    return {EFFECT: ate, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_att(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    mu_1, mu_0 = compute_weighted_outcomes_treated(A, Y, ps, stabilized=stabilized)
    att = mu_1 - mu_0
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_risk_ratio_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    """
    Computes the Relative Risk for the Treated (RRT) using IPW.
    """
    mu_1, mu_0 = compute_weighted_outcomes_treated(A, Y, ps, stabilized=stabilized)
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
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> Tuple[float, float]:
    """
    Computes E[Y(1)] and E[Y(0)] for the ATE using the simple Horvitz-Thompson estimator,
    with explicit checks for empty groups.
    """
    W = compute_ipw_weights(A, ps, weight_type="ATE", stabilized=stabilized)

    # --- Calculate for Treated Group (mu_1) ---
    if A.sum() > 0:
        mu_1 = (W * A * Y).mean()
    else:
        warnings.warn("No subjects in the treated group. mu_1 is NaN.", RuntimeWarning)
        mu_1 = np.nan

    # --- Calculate for Control Group (mu_0) ---
    if (1 - A).sum() > 0:
        mu_0 = (W * (1 - A) * Y).mean()
    else:
        warnings.warn("No subjects in the control group. mu_0 is NaN.", RuntimeWarning)
        mu_0 = np.nan

    return mu_1, mu_0


def compute_weighted_outcomes_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> Tuple[float, float]:
    """
    Computes E[Y(1)|A=1] and E[Y(0)|A=1] for the ATT, with explicit checks for empty groups.
    """
    W = compute_ipw_weights(A, ps, weight_type="ATT", stabilized=stabilized)

    # --- Factual Outcome for the Treated (mu_1) ---
    num_treated = A.sum()
    if num_treated > 0:
        mu_1 = Y[A == 1].mean()  # no asjustment for treated
    else:
        warnings.warn(
            "No subjects in the treated group for ATT. mu_1 is NaN.", RuntimeWarning
        )
        mu_1 = np.nan

    # --- Counterfactual Outcome for the Treated (mu_0) ---
    if num_treated > 0 and (1 - A).sum() > 0:
        mu_0 = (W * (1 - A) * Y).sum() / num_treated
    else:
        # mu_0 is NaN if there are no treated (target population) or no controls (source population)
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
    stabilized: bool = False,
) -> np.ndarray:
    """
    Compute IPW weights for ATE or ATT with optional stabilization.
    """
    if weight_type == "ATE":
        if stabilized:
            pi = A.mean()
            weight_treated = pi / ps
            weight_control = (1 - pi) / (1 - ps)
        else:
            weight_treated = 1 / ps
            weight_control = 1 / (1 - ps)
        return A * weight_treated + (1 - A) * weight_control

    elif weight_type == "ATT":
        weight_treated = np.ones_like(A, dtype=float)
        weight_control = ps / (1 - ps)
        if stabilized:
            pi = A.mean()
            if pi > 0:
                stabilization_factor = (1 - pi) / pi
                weight_control *= stabilization_factor
        return A * weight_treated + (1 - A) * weight_control

    else:
        raise ValueError("weight_type must be 'ATE' or 'ATT'")
