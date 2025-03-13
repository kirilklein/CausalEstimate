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
"""

import warnings
from typing import Tuple

import numpy as np

from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated


def compute_ipw_risk_ratio(A: np.ndarray, Y: np.ndarray, ps: np.ndarray) -> dict:
    """
    Relative Risk
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu_1, mu_0 = compute_mean_potential_outcomes(A, Y, ps)
    rr = mu_1 / mu_0
    return {EFFECT: rr, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_ate(A: np.ndarray, Y: np.ndarray, ps: np.ndarray) -> dict:
    """
    Average Treatment Effect
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu1, mu0 = compute_mean_potential_outcomes(A, Y, ps)
    ate = mu1 - mu0
    return {EFFECT: ate, EFFECT_treated: mu1, EFFECT_untreated: mu0}


def compute_mean_potential_outcomes(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray
) -> Tuple[float, float]:
    """
    Compute E[Y|A=1] and E[Y|A=0] for Y=0/1
    """
    mu_1 = (A * Y / ps).mean()
    mu_0 = ((1 - A) * Y / (1 - ps)).mean()
    return mu_1, mu_0


def compute_ipw_ate_stabilized(A: np.ndarray, Y: np.ndarray, ps: np.ndarray) -> dict:
    """
    Given by Austin (2016)
    Average Treatment Effect with stabilized weights.
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    W = compute_stabilized_ate_weights(A, ps)
    Y1_weighed = (W * A * Y).mean()
    Y0_weighed = (W * (1 - A) * Y).mean()
    ate = Y1_weighed - Y0_weighed
    return {EFFECT: ate, EFFECT_treated: Y1_weighed, EFFECT_untreated: Y0_weighed}


def compute_ipw_att(A: np.ndarray, Y: np.ndarray, ps: np.ndarray) -> dict:
    """
    Average Treatment Effect on the Treated with stabilized weights.
    Reifeis et. al. (2022).
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu_1, mu_0 = compute_mean_potential_outcomes_treated(A, Y, ps)
    att = mu_1 - mu_0
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_ipw_risk_ratio_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray
) -> dict:
    """
    Relative Risk of the Treated with stabilized weights. Reifeis et. al. (2022)
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu_1, mu_0 = compute_mean_potential_outcomes_treated(A, Y, ps)
    if mu_0 == 0:
        warnings.warn("mu_0 is 0, returning inf", RuntimeWarning)
        return {EFFECT: np.inf, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}
    rr = mu_1 / mu_0
    return {EFFECT: rr, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_mean_potential_outcomes_treated(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray
) -> Tuple[float, float]:
    """
    Compute E[Y|A=1] for Y=0/1
    """
    W = compute_stabilized_att_weights(A, ps)
    mu_1 = (W * A * Y).sum() / (W * A).sum()
    mu_0 = (W * (1 - A) * Y).sum() / (W * (1 - A)).sum()
    return mu_1, mu_0


def compute_stabilized_ate_weights(A: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Compute the (stabilized) weights for the ATE estimator
    Austin (2016)
    """
    weight_treated = A.mean() * A / ps
    weight_control = (1 - A).mean() * (1 - A) / (1 - ps)
    return weight_treated + weight_control


def compute_stabilized_att_weights(A: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Compute the (stabilized) weights for the ATT estimator
    As given in the web appendix of Reifeis et. al. (2022)
    """
    h = ps / (1 - ps)
    return A + (1 - A) * h
