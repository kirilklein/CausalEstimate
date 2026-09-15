"""
TMLE estimators for the ATE and the Risk Ratio.

Both call `target_outcome_models`, which runs one weighted intercept-only
fluctuation per arm and returns the targeted predictions together with the
weights it used. The estimands differ only in the choice of weights and in
how the two targeted arm means are combined.

The ATT estimator lives in a separate module and uses the same function
with effect_type="ATT".
"""

from typing import Optional

import numpy as np

from CausalEstimate.estimators.functional.utils import (
    check_score_equations,
    compute_initial_effect,
    safe_ratio,
    target_outcome_models,
)
from CausalEstimate.estimators.functional.variance import compute_ci
from CausalEstimate.utils.constants import (
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
)


def compute_tmle_ate(
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
    Estimate the ATE using TMLE, with optional per-arm weight clipping.

    `Yhat` is accepted and ignored, for backwards compatibility only: each
    arm is now fluctuated from its own predictions, so the prediction at the
    observed treatment is derived rather than supplied.
    """

    result = target_outcome_models(
        A,
        Y,
        ps,
        Y1_hat,
        Y0_hat,
        effect_type="ATE",
        clip_percentile=clip_percentile,
        eps=eps,
    )
    check_score_equations(result, Y)

    Q_star_1_m = float(result.Q_star_1.mean())
    Q_star_0_m = float(result.Q_star_0.mean())
    ate = Q_star_1_m - Q_star_0_m

    ci_results = compute_ci(
        effect_type="ATE",
        psi=ate,
        Q_star_1=result.Q_star_1,
        Q_star_0=result.Q_star_0,
        Y=Y,
        A=A,
        Yhat_star=result.Yhat_star,
        H=result.H,
    )

    return {
        EFFECT: ate,
        EFFECT_treated: Q_star_1_m,
        EFFECT_untreated: Q_star_0_m,
        **compute_initial_effect(Y1_hat, Y0_hat, result.Q_star_1, result.Q_star_0),
        **ci_results,
    }


def compute_tmle_rr(
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
    Estimate the Risk Ratio using TMLE, with optional per-arm weight clipping.

    Identical to the ATE apart from the final combination step: each arm mean
    is targeted separately, which is what a ratio requires. A single
    fluctuation targeting the difference leaves each arm mean individually
    biased, and those biases do not cancel in a ratio.
    """

    result = target_outcome_models(
        A,
        Y,
        ps,
        Y1_hat,
        Y0_hat,
        effect_type="RR",
        clip_percentile=clip_percentile,
        eps=eps,
    )
    check_score_equations(result, Y)

    Q_star_1_m = float(result.Q_star_1.mean())
    Q_star_0_m = float(result.Q_star_0.mean())
    rr = safe_ratio(Q_star_1_m, Q_star_0_m, label="Risk ratio")

    ci_results = compute_ci(
        effect_type="RR",
        psi=rr,
        Q_star_1=result.Q_star_1,
        Q_star_0=result.Q_star_0,
        Y=Y,
        A=A,
        Yhat_star=result.Yhat_star,
        w1=result.w1,
        w0=result.w0,
    )

    return {
        EFFECT: rr,
        EFFECT_treated: Q_star_1_m,
        EFFECT_untreated: Q_star_0_m,
        **compute_initial_effect(
            Y1_hat, Y0_hat, result.Q_star_1, result.Q_star_0, rr=True
        ),
        **ci_results,
    }
