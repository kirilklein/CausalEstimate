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
Hernán MA, Robins JM (2020). Causal Inference: What If.
Boca Raton: Chapman & Hall/CRC. (Chapter 12)
https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

"""

import warnings
from typing import Tuple, Literal

import numpy as np

from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated

# --- Core Effect Calculation Functions ---


def compute_ipw_risk_ratio(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    """
    Computes the Relative Risk using IPW.
    """
    mu_1, mu_0 = compute_mean_potential_outcomes(A, Y, ps, stabilized=stabilized)
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
    """
    Computes the Average Treatment Effect (ATE) using IPW.
    """
    mu_1, mu_0 = compute_mean_potential_outcomes(A, Y, ps, stabilized=stabilized)
    ate = mu_1 - mu_0
    return {EFFECT: ate, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_att(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    """
    Computes the Average Treatment Effect on the Treated (ATT) using IPW.
    """
    mu_1, mu_0 = compute_mean_potential_outcomes_treated(
        A, Y, ps, stabilized=stabilized
    )
    att = mu_1 - mu_0
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_risk_ratio_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> dict:
    """
    Computes the Relative Risk for the Treated (RRT) using IPW.
    """
    mu_1, mu_0 = compute_mean_potential_outcomes_treated(
        A, Y, ps, stabilized=stabilized
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


# --- Potential Outcome Mean Estimators (Refactored) ---


def compute_mean_potential_outcomes(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> Tuple[float, float]:
    """
    Computes E[Y(1)] and E[Y(0)] for the ATE using a consistent weighted mean formula.
    Handles edge cases where one treatment group is empty.
    """
    W = compute_ipw_weights(A, ps, weight_type="ATE", stabilized=stabilized)

    # --- Calculate for Treated Group (mu_1) ---
    sum_w_treated = (W * A).sum()
    if sum_w_treated == 0:
        warnings.warn(
            "No subjects in the treated group (or sum of weights is zero). mu_1 is NaN.",
            RuntimeWarning,
        )
        mu_1 = np.nan
    else:
        mu_1 = (W * A * Y).sum() / sum_w_treated

    # --- Calculate for Control Group (mu_0) ---
    sum_w_control = (W * (1 - A)).sum()
    if sum_w_control == 0:
        warnings.warn(
            "No subjects in the control group (or sum of weights is zero). mu_0 is NaN.",
            RuntimeWarning,
        )
        mu_0 = np.nan
    else:
        mu_0 = (W * (1 - A) * Y).sum() / sum_w_control

    return mu_1, mu_0


def compute_mean_potential_outcomes_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, stabilized: bool = False
) -> Tuple[float, float]:
    """
    Computes E[Y(1)|A=1] and E[Y(0)|A=1] for the ATT using a consistent weighted mean formula.
    Handles edge cases where one treatment group is empty.
    """
    W = compute_ipw_weights(A, ps, weight_type="ATT", stabilized=stabilized)

    # --- Calculate for Treated Group (mu_1) ---
    num_treated = A.sum()
    if num_treated == 0:
        warnings.warn(
            "No subjects in the treated group for ATT (or sum of weights is zero). mu_1 is NaN.",
            RuntimeWarning,
        )
        mu_1 = np.nan
    else:
        # For ATT, the weight for the treated is 1, so this simplifies to mean(Y[A==1])
        mu_1 = Y[A == 1].mean()

    # --- Calculate for Control Group (mu_0) ---
    sum_w_control = (W * (1 - A)).sum()
    if sum_w_control == 0:
        warnings.warn(
            "No subjects in the control group for ATT (or sum of weights is zero). mu_0 is NaN.",
            RuntimeWarning,
        )
        mu_0 = np.nan
    else:
        # For ATT, controls are weighted to look like the treated population
        mu_0 = (W * (1 - A) * Y).sum() / sum_w_control

    return mu_1, mu_0


# --- Centralized Weight Calculation Function (Corrected and Finalized) ---


def compute_ipw_weights(
    A: np.ndarray,
    ps: np.ndarray,
    weight_type: Literal["ATE", "ATT"] = "ATE",
    stabilized: bool = False,
) -> np.ndarray:
    """
    Compute IPW weights for ATE or ATT with optional stabilization.

    Args:
        A: Treatment assignment vector (binary).
        ps: Propensity score vector.
        weight_type: The type of effect, either 'ATE' or 'ATT'.
        stabilized: If True, computes stabilized weights.

    Returns:
        A numpy array containing the IPW weights for each observation.

    Formulas:
        pi = P(A=1) estimated as mean(A)

        ATE:
            - Unstabilized: w = A/ps + (1-A)/(1-ps)
            - Stabilized:   w = A*(pi/ps) + (1-A)*((1-pi)/(1-ps))

        ATT:
            - Unstabilized: w = A + (1-A)*(ps/(1-ps))
            - Stabilized:   w = A + (1-A)*(ps/(1-ps)) * ((1-pi)/pi)
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
        # Weight for treated is always 1
        weight_treated = np.ones_like(A, dtype=float)

        # Weight for control
        weight_control = ps / (1 - ps)
        if stabilized:
            pi = A.mean()
            # Avoid division by zero if there are no treated subjects
            if pi > 0:
                stabilization_factor = (1 - pi) / pi
                weight_control *= stabilization_factor

        return A * weight_treated + (1 - A) * weight_control

    else:
        raise ValueError("weight_type must be 'ATE' or 'ATT'")
