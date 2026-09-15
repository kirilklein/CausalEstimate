"""
TMLE estimator for the ATT.

Like the ATE/RR estimators in `tmle.py`, this calls `target_outcome_models`,
which runs one weighted intercept-only fluctuation per arm and returns the
targeted predictions together with the weights it used. Only the weights
(effect_type="ATT": w1 = A/p_treated, w0 = (1-A) ps / (p_treated (1-ps))) and
the final combination step -- an average over the treated only -- differ.

The implementation is largely based on the following reference:
Van der Laan MJ, Rose S. Targeted learning: causal inference for observational and experimental data. Springer; New York: 2011. Specifically, Chapter 8 for the ATT TMLE.
But slightly modified for simpler implementation, following advice from: https://stats.stackexchange.com/questions/520472/can-targeted-maximum-likelihood-estimation-find-the-average-treatment-effect-on/534018#534018
"""

from typing import Optional

import numpy as np

from CausalEstimate.estimators.functional.utils import (
    check_score_equations,
    compute_initial_effect,
    target_outcome_models,
)
from CausalEstimate.estimators.functional.variance import compute_ci
from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated


def compute_tmle_att(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: Optional[np.ndarray] = None,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Estimate the Average Treatment Effect on the Treated (ATT) using TMLE.

    `Yhat` is accepted and ignored, for backwards compatibility only: each
    arm is now fluctuated from its own predictions, so the prediction at the
    observed treatment is derived rather than supplied.
    """
    treated_mask = A == 1
    if not np.any(treated_mask):
        # No treated subjects: the ATT is not identified.
        return {EFFECT: np.nan, EFFECT_treated: np.nan, EFFECT_untreated: np.nan}

    result = target_outcome_models(
        A,
        Y,
        ps,
        Y1_hat,
        Y0_hat,
        effect_type="ATT",
        clip_percentile=clip_percentile,
        eps=eps,
    )
    check_score_equations(result, Y)

    Q_star_1_m = float(result.Q_star_1[treated_mask].mean())
    Q_star_0_m = float(result.Q_star_0[treated_mask].mean())
    psi = Q_star_1_m - Q_star_0_m

    ci_results = compute_ci(
        effect_type="ATT",
        psi=psi,
        Q_star_1=result.Q_star_1,
        Q_star_0=result.Q_star_0,
        Y=Y,
        A=A,
        Yhat_star=result.Yhat_star,
        H=result.H,
    )

    return {
        EFFECT: psi,
        EFFECT_treated: Q_star_1_m,
        EFFECT_untreated: Q_star_0_m,
        **compute_initial_effect(Y1_hat, Y0_hat, result.Q_star_1, result.Q_star_0),
        **ci_results,
    }
