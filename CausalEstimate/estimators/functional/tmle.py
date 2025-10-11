import warnings
from typing import Tuple

import numpy as np
from scipy.special import expit, logit

from CausalEstimate.estimators.functional.utils import (
    compute_clever_covariate_ate,
    compute_initial_effect,
    estimate_fluctuation_parameter,
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
    Yhat: np.ndarray,
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> dict:
    """
    Estimate the ATE using TMLE, with optional weight clipping.
    """
    Q_star_1, Q_star_0, Yhat_star, H = compute_estimates(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, clip_percentile=clip_percentile, eps=eps
    )
    ate = (Q_star_1 - Q_star_0).mean()

    ci_results = compute_ci(
        effect_type="ATE",
        psi=ate,
        Q_star_1=Q_star_1,
        Q_star_0=Q_star_0,
        Y=Y,
        A=A,
        ps=ps,
        Yhat_star=Yhat_star,
        H=H,
    )

    return {
        EFFECT: ate,
        EFFECT_treated: Q_star_1.mean(),
        EFFECT_untreated: Q_star_0.mean(),
        **compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0),
        **ci_results,
    }


def compute_tmle_rr(
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
    Estimate the Risk Ratio using TMLE, with optional weight clipping.
    """
    Q_star_1, Q_star_0, Yhat_star, H = compute_estimates(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, clip_percentile=clip_percentile, eps=eps
    )
    Q_star_1_m = Q_star_1.mean()
    Q_star_0_m = Q_star_0.mean()

    if np.isclose(Q_star_0_m, 0, atol=1e-8):
        warnings.warn(
            "Mean of Q_star_0 is 0, returning inf for Risk Ratio.", RuntimeWarning
        )
        rr = np.inf
    else:
        rr = Q_star_1_m / Q_star_0_m

    if rr > 1e5:
        warnings.warn(
            "Risk ratio is unrealistically large, returning inf.", RuntimeWarning
        )
        rr = np.inf

    ci_results = compute_ci(
        effect_type="RR",
        psi=rr,
        Q_star_1=Q_star_1,
        Q_star_0=Q_star_0,
        Y=Y,
        A=A,
        ps=ps,
        Yhat_star=Yhat_star,
        H=H,
    )

    return {
        EFFECT: rr,
        EFFECT_treated: Q_star_1_m,
        EFFECT_untreated: Q_star_0_m,
        **compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0, rr=True),
        **ci_results,
    }


def compute_estimates(
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
    Compute updated outcome estimates using TMLE targeting step.
    Returns:
        Q_star_1: Updated outcome estimates under treatment
        Q_star_0: Updated outcome estimates under control
        Yhat_star: Targeted predictions Q*(A,W)
        H: Clever covariate
    """
    H = compute_clever_covariate_ate(A, ps, clip_percentile=clip_percentile, eps=eps)
    epsilon = estimate_fluctuation_parameter(H, Y, Yhat)
    Q_star_1, Q_star_0 = update_estimates(ps, Y0_hat, Y1_hat, epsilon, eps=eps)

    Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
    Yhat_star = expit(logit(Yhat_clipped) + epsilon * H)

    return Q_star_1, Q_star_0, Yhat_star, H


def update_estimates(
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    epsilon: float,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the initial outcome estimates using the fluctuation parameter.
    Returns:
        Q_star_1: Updated outcome estimates under treatment
        Q_star_0: Updated outcome estimates under control
    """
    H1 = 1.0 / (ps + eps)
    H0 = -1.0 / (1.0 - ps + eps)

    Q_star_1 = expit(logit(np.clip(Y1_hat, eps, 1 - eps)) + epsilon * H1)
    Q_star_0 = expit(logit(np.clip(Y0_hat, eps, 1 - eps)) + epsilon * H0)

    return Q_star_1, Q_star_0
