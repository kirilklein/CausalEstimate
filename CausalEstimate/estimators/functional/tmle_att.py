"""
TMLE estimators for a single-arm subpopulation: the ATT and the RRT over the
treated, and the ATC over the controls.

Like the ATE/RR estimators in `tmle.py`, these call `target_outcome_models`,
which runs one weighted intercept-only fluctuation per arm and returns the
targeted predictions together with the weights it used. Only the weights
(w1 = A/p_treated, w0 = (1-A) ps / (p_treated (1-ps))) and the final
combination step -- an average over the treated only -- differ.

The ATT and the RRT share those weights and differ only in how the two
targeted arm means are combined, exactly as the ATE and the RR do. The ATC
is the ATT with the arms swapped (see `att_result_as_atc`).

The implementation is largely based on the following reference:
Van der Laan MJ, Rose S. Targeted learning: causal inference for observational and experimental data. Springer; New York: 2011. Specifically, Chapter 8 for the ATT TMLE.
But slightly modified for simpler implementation, following advice from: https://stats.stackexchange.com/questions/520472/can-targeted-maximum-likelihood-estimation-find-the-average-treatment-effect-on/534018#534018
"""

import numpy as np

from CausalEstimate.estimators.functional.utils import (
    att_result_as_atc,
    check_score_equations,
    compute_initial_effect,
    safe_ratio,
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
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Estimate the Average Treatment Effect on the Treated (ATT) using TMLE.

    Each arm is fluctuated from its own predictions, so the prediction
    at the observed treatment is derived rather than supplied.
    """
    treated_mask = A == 1
    if not np.any(treated_mask) or np.all(treated_mask):
        # Without both arms the ATT is not identified: with no controls the
        # control fluctuation is skipped and Q_star_0 is just Y0_hat, which
        # would otherwise yield a finite estimate with a spuriously tight CI.
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
        **compute_initial_effect(
            Y1_hat, Y0_hat, result.Q_star_1, result.Q_star_0, mask=treated_mask
        ),
        **ci_results,
    }


def compute_tmle_atc(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Estimate the Average Treatment Effect on the Controls (ATC) using TMLE, as
    the ATT with the arms swapped (see `att_result_as_atc`).
    """
    return att_result_as_atc(
        compute_tmle_att(
            1 - A,
            Y,
            1 - ps,
            Y1_hat,
            Y0_hat,
            clip_percentile=clip_percentile,
            eps=eps,
        )
    )


def compute_tmle_rrt(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Estimate the Risk Ratio in the Treated (RRT) using TMLE.

    Identical to the ATT apart from the final combination step: the two
    targeted treated-arm means are ratioed rather than differenced. As for the
    RR, each arm mean is targeted separately, since a single fluctuation
    targeting the difference leaves each mean individually biased and those
    biases do not cancel in a ratio.
    """
    treated_mask = A == 1
    if not np.any(treated_mask) or np.all(treated_mask):
        # Without both arms the RRT is not identified: with no controls the
        # control fluctuation is skipped and Q_star_0 is just Y0_hat, which
        # would otherwise yield a finite ratio with a spuriously tight CI.
        return {EFFECT: np.nan, EFFECT_treated: np.nan, EFFECT_untreated: np.nan}

    result = target_outcome_models(
        A,
        Y,
        ps,
        Y1_hat,
        Y0_hat,
        effect_type="RRT",
        clip_percentile=clip_percentile,
        eps=eps,
    )
    check_score_equations(result, Y)

    Q_star_1_m = float(result.Q_star_1[treated_mask].mean())
    Q_star_0_m = float(result.Q_star_0[treated_mask].mean())
    rrt = safe_ratio(Q_star_1_m, Q_star_0_m, label="Risk ratio in the treated")

    ci_results = compute_ci(
        effect_type="RRT",
        psi=rrt,
        Q_star_1=result.Q_star_1,
        Q_star_0=result.Q_star_0,
        Y=Y,
        A=A,
        Yhat_star=result.Yhat_star,
        w1=result.w1,
        w0=result.w0,
        eps=eps,
    )

    return {
        EFFECT: rrt,
        EFFECT_treated: Q_star_1_m,
        EFFECT_untreated: Q_star_0_m,
        **compute_initial_effect(
            Y1_hat,
            Y0_hat,
            result.Q_star_1,
            result.Q_star_0,
            rr=True,
            mask=treated_mask,
        ),
        **ci_results,
    }
