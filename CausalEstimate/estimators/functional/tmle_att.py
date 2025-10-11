"""
The implementation is largely based on the following reference:
Van der Laan MJ, Rose S. Targeted learning: causal inference for observational and experimental data. Springer; New York: 2011. Specifically, Chapter 8 for the ATT TMLE.
But slightly modified for simpler implementation, following advice from: https://stats.stackexchange.com/questions/520472/can-targeted-maximum-likelihood-estimation-find-the-average-treatment-effect-on/534018#534018
"""

from typing import Tuple

import numpy as np
from scipy.special import expit, logit

from CausalEstimate.estimators.functional.utils import (
    compute_clever_covariate_att,
    compute_initial_effect,
    estimate_fluctuation_parameter,
)
from CausalEstimate.estimators.functional.variance import compute_ci
from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated


def compute_estimates_att(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute updated outcome estimates for ATT using a one-step TMLE targeting step.
    """
    H = compute_clever_covariate_att(A, ps, clip_percentile=clip_percentile, eps=eps)
    epsilon = estimate_fluctuation_parameter(H, Y, Yhat)

    p_treated = np.mean(A == 1)
    if p_treated == 0:
        Yhat_star = Yhat.copy()  # No update if no treated
        return Y1_hat, Y0_hat, Yhat_star, H

    # Update terms
    update_term_1 = epsilon * (1.0 / (p_treated + eps))
    weight_component = ps / (p_treated * (1 - ps) + eps)

    if clip_percentile < 1:
        control_mask: np.ndarray = A == 0
        if control_mask.sum() > 0:
            control_weights = weight_component[control_mask]
            threshold = np.percentile(control_weights, clip_percentile * 100)
            weight_component = np.clip(weight_component, a_min=None, a_max=threshold)

    update_term_0 = -epsilon * weight_component

    # Apply updates
    Q_star_1 = expit(logit(np.clip(Y1_hat, eps, 1 - eps)) + update_term_1)
    Q_star_0 = expit(logit(np.clip(Y0_hat, eps, 1 - eps)) + update_term_0)

    Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
    Yhat_star = expit(logit(Yhat_clipped) + epsilon * H)

    return Q_star_1, Q_star_0, Yhat_star, H


def compute_tmle_att(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Estimate the Average Treatment Effect on the Treated (ATT) using TMLE.
    """
    Q_star_1, Q_star_0, Yhat_star, H = compute_estimates_att(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, clip_percentile=clip_percentile, eps=eps
    )

    treated_mask = A == 1
    if not np.any(treated_mask):
        # Handle case with no treated subjects
        return {EFFECT: np.nan, EFFECT_treated: np.nan, EFFECT_untreated: np.nan}

    psi = np.mean(Q_star_1[treated_mask] - Q_star_0[treated_mask])

    ci_results = compute_ci(
        effect_type="ATT",
        psi=psi,
        Q_star_1=Q_star_1,
        Q_star_0=Q_star_0,
        Y=Y,
        A=A,
        ps=ps,
        Yhat_star=Yhat_star,
        H=H,
    )

    return {
        EFFECT: psi,
        EFFECT_treated: np.mean(Q_star_1[treated_mask]),
        EFFECT_untreated: np.mean(Q_star_0[treated_mask]),
        **compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0),
        **ci_results,
    }
