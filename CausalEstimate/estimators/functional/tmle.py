import warnings
from typing import Tuple, Optional

import numpy as np
from scipy.special import expit, logit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM

from CausalEstimate.estimators.functional.utils import compute_initial_effect
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
    stabilized: bool = False,  # New parameter
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
    stabilized: bool = False,  # New parameter
) -> dict:
    """
    Estimate the Risk Ratio using TMLE, with optional weight stabilization.
    """
    Q_star_1, Q_star_0 = compute_estimates(
        A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=stabilized
    )
    Q_star_1_m = Q_star_1.mean()
    Q_star_0_m = Q_star_0.mean()

    if Q_star_0_m == 0:
        warnings.warn(
            "Mean of Q_star_0 is 0, returning inf for Risk Ratio.", RuntimeWarning
        )
        rr = np.inf
    else:
        rr = Q_star_1_m / Q_star_0_m

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
    epsilon = estimate_fluctuation_parameter(A, Y, ps, Yhat, stabilized=stabilized)

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


def estimate_fluctuation_parameter(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Yhat: np.ndarray,
    stabilized: bool = False,
) -> float:
    """
    Estimate the fluctuation parameter epsilon using a logistic regression model.
    """
    if stabilized:
        pi = A.mean()
        # Stabilized clever covariate
        H = A * pi / ps - (1 - A) * (1 - pi) / (1 - ps)
    else:
        # Unstabilized clever covariate
        H = A / ps - (1 - A) / (1 - ps)

    if np.any(np.abs(H) > 100):
        warnings.warn(
            "Extreme values detected in clever covariate H. "
            "This may indicate issues with propensity scores near 0 or 1.",
            RuntimeWarning,
        )

    offset = logit(Yhat)
    model = GLM(Y, H, family=Binomial(), offset=offset).fit()

    # model.params is a one-element array containing epsilon
    return np.asarray(model.params)[0]
