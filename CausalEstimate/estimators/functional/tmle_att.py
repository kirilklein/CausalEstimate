"""
The implementation is largely based on the following reference:
Van der Laan MJ, Rose S. Targeted learning: causal inference for observational and experimental data. Springer; New York: 2011. Specifically, Chapter 8 for the ATT TMLE.
But slightly modified for simpler implementation, following advice from: https://stats.stackexchange.com/questions/520472/can-targeted-maximum-likelihood-estimation-find-the-average-treatment-effect-on/534018#534018
"""

from typing import Tuple

import numpy as np
from scipy.special import expit, logit

from CausalEstimate.estimators.functional.utils import (
    compute_initial_effect,
    compute_clever_covariate_att,
    estimate_fluctuation_parameter,
)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute updated outcome estimates for ATT using a one-step TMLE targeting step.
    """
    # Estimate the fluctuation parameter epsilon using a logistic regression:
    H = compute_clever_covariate_att(A, ps, clip_percentile=clip_percentile, eps=eps)
    epsilon = estimate_fluctuation_parameter(H, Y, Yhat)

    # --- Step 2: Define the CORRECT, separate update terms ---
    # This is the part that was incorrect in your new code. We revert to the logic
    # from your old implementation.
    p_treated = np.mean(A == 1)
    if (
        p_treated == 0
    ):  # Should be caught by compute_clever_covariate_att but good practice
        return Y1_hat, Y0_hat

    # The update term for the potential outcome under treatment, Q(1,W).
    # This is a scalar value applied to everyone's Y1_hat.
    update_term_1 = epsilon * (1.0 / (p_treated + eps))

    # The update term for the potential outcome under control, Q(0,W).
    # This is a vector of values applied to everyone's Y0_hat.
    # We must re-calculate the weight component here.
    # For theoretical consistency, if ps were clipped to find H, they should be clipped here too.

    weight_component = ps / (p_treated * (1 - ps) + eps)

    if clip_percentile < 1:
        control_mask: np.ndarray = A == 0
        if control_mask.sum() > 0:
            control_weights = weight_component[control_mask]
            threshold = np.percentile(control_weights, clip_percentile * 100)
            # Clip the component for ALL subjects based on the threshold from controls
            weight_component = np.clip(weight_component, a_min=None, a_max=threshold)

    update_term_0 = -epsilon * weight_component

    # --- Step 3: Apply the separate updates to the potential outcome models ---
    Q_star_1 = expit(logit(Y1_hat) + update_term_1)
    Q_star_0 = expit(logit(Y0_hat) + update_term_0)

    return Q_star_1, Q_star_0


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
    Estimate the Average Treatment Effect on the Treated (ATT) using TMLE,
    with optional clipping for the control group.
    eps: float = 1e-9,
        Guard against division by zero
    """
    Q_star_1, Q_star_0 = compute_estimates_att(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, clip_percentile=clip_percentile, eps=eps
    )

    # The final ATT parameter is the mean difference within the treated population
    psi = np.mean(Q_star_1[A == 1] - Q_star_0[A == 1])

    return {
        EFFECT: psi,
        # For clarity, return the mean of the updated predictions
        EFFECT_treated: np.mean(Q_star_1[A == 1]),
        EFFECT_untreated: np.mean(Q_star_0[A == 1]),
        **compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0),
    }
