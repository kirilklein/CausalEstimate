import warnings
from typing import Tuple, Optional

import numpy as np
from scipy.special import expit, logit


from CausalEstimate.estimators.functional.utils import (
    compute_initial_effect,
    compute_clever_covariate_ate,
    estimate_fluctuation_parameter,
)
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
    stabilized: bool = False,
) -> dict:
    """
    Estimate the ATE using TMLE, with optional weight stabilization.
    """
    Q_star_1, Q_star_0 = compute_estimates(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=stabilized
    )
    ate = (Q_star_1 - Q_star_0).mean()

    return {
        EFFECT: ate,
        EFFECT_treated: Q_star_1.mean(),  # Return mean of predictions
        EFFECT_untreated: Q_star_0.mean(),  # Return mean of predictions
        **compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0),
    }


def compute_tmle_rr(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
    stabilized: bool = False,
) -> dict:
    """
    Estimate the Risk Ratio using TMLE, with optional weight stabilization.
    """
    Q_star_1, Q_star_0 = compute_estimates(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=stabilized
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

    return {
        EFFECT: rr,
        EFFECT_treated: Q_star_1_m,
        EFFECT_untreated: Q_star_0_m,
        **compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0, rr=True),
    }


def compute_estimates(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
    stabilized: bool = False,  # New parameter
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute updated outcome estimates using TMLE targeting step.
    """
    H = compute_clever_covariate_ate(A, ps, stabilized=stabilized)
    epsilon = estimate_fluctuation_parameter(H, Y, Yhat)

    pi = A.mean() if stabilized else None
    Q_star_1, Q_star_0 = update_estimates(ps, Y0_hat, Y1_hat, epsilon, pi=pi)

    return Q_star_1, Q_star_0


def update_estimates(
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    epsilon: float,
    pi: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the initial outcome estimates using the fluctuation parameter.
    If pi is provided, uses stabilized clever covariates.
    """
    if pi is not None:
        # Stabilized clever covariates
        H1 = pi / ps
        H0 = -(1.0 - pi) / (1.0 - ps)
    else:
        # Unstabilized clever covariates
        H1 = 1.0 / ps
        H0 = -1.0 / (1.0 - ps)

    # Update initial estimates with targeting step
    Q_star_1 = expit(logit(Y1_hat) + epsilon * H1)
    Q_star_0 = expit(logit(Y0_hat) + epsilon * H0)

    return Q_star_1, Q_star_0
