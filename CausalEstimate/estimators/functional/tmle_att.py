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
    stabilized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute updated outcome estimates for ATT using a one-step TMLE targeting step.
    """
    # Estimate the fluctuation parameter epsilon using a logistic regression:
    H = compute_clever_covariate_att(A, ps, stabilized=stabilized)
    epsilon = estimate_fluctuation_parameter(H, Y, Yhat)

    p_treated = np.mean(A == 1)

    # The update term for the treated group is always the same
    update_term_1 = epsilon * (1.0 / p_treated)

    # The update term for the control group depends on stabilization
    if stabilized:
        # Stabilized update term for controls
        update_term_0 = -epsilon * (ps * (1 - p_treated) / (p_treated * (1 - ps)))
    else:
        # Unstabilized update term for controls
        update_term_0 = -epsilon * (ps / (p_treated * (1 - ps)))

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
    stabilized: bool = False,
) -> dict:
    """
    Estimate the Average Treatment Effect on the Treated (ATT) using TMLE,
    with optional weight stabilization for the control group.
    """
    Q_star_1, Q_star_0 = compute_estimates_att(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=stabilized
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
