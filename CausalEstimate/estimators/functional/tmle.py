import warnings
from typing import Tuple

import numpy as np
from scipy.special import expit, logit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM

from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated


def compute_tmle_ate(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
) -> dict:
    """
    Estimate the average treatment effect using the targeted maximum likelihood estimation (TMLE) method.

    Parameters:
    -----------
    A: array-like
        Treatment assignment (0 or 1)
    Y: array-like
        Binary outcome
    ps: array-like
        Propensity score P(A=1|X)
    Y0_hat: array-like
        Initial outcome prediction for control group P(Y|A=0,X)
    Y1_hat: array-like
        Initial outcome prediction for treatment group P(Y|A=1,X)
    Yhat: array-like
        Combined outcome prediction (A*Y1_hat + (1-A)*Y0_hat)

    Returns:
    --------
    float: Average treatment effect estimate
    """
    Q_star_1, Q_star_0 = compute_estimates(A, Y, ps, Y0_hat, Y1_hat, Yhat)
    ate = (Q_star_1 - Q_star_0).mean()
    return {EFFECT: ate, EFFECT_treated: Q_star_1, EFFECT_untreated: Q_star_0}


def compute_tmle_rr(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
) -> dict:
    """
    Estimate the risk ratio using the targeted maximum likelihood estimation (TMLE) method.

    Parameters:
    -----------
    A: array-like
        Treatment assignment (0 or 1)
    Y: array-like
        Binary outcome
    ps: array-like
        Propensity score P(A=1|X)
    Y0_hat: array-like
        Initial outcome prediction for control group P(Y|A=0,X)
    Y1_hat: array-like
        Initial outcome prediction for treatment group P(Y|A=1,X)
    Yhat: array-like
        Combined outcome prediction (A*Y1_hat + (1-A)*Y0_hat)

    Returns:
    --------
    float: Risk ratio estimate
    """
    Q_star_1, Q_star_0 = compute_estimates(A, Y, ps, Y0_hat, Y1_hat, Yhat)
    Q_star_0_m = Q_star_0.mean()
    if Q_star_0_m == 0:
        warnings.warn("Q_star_0 is 0, returning inf", RuntimeWarning)
        return np.inf
    rr = Q_star_1.mean() / Q_star_0_m
    return {EFFECT: rr, EFFECT_treated: Q_star_1.mean(), EFFECT_untreated: Q_star_0_m}


def compute_estimates(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute updated outcome estimates using TMLE targeting step.

    Parameters:
    -----------
    A: array-like
        Treatment assignment (0 or 1)
    Y: array-like
        Binary outcome
    ps: array-like
        Propensity score P(A=1|X)
    Y0_hat: array-like
        Initial outcome prediction for control group P(Y|A=0,X)
    Y1_hat: array-like
        Initial outcome prediction for treatment group P(Y|A=1,X)
    Yhat: array-like
        Combined outcome prediction (A*Y1_hat + (1-A)*Y0_hat)

    Returns:
    --------
    tuple: (Q_star_1, Q_star_0)
        Updated outcome predictions for treatment and control groups
    """
    epsilon = estimate_fluctuation_parameter(A, Y, ps, Yhat)
    Q_star_1, Q_star_0 = update_estimates(ps, Y0_hat, Y1_hat, epsilon)
    return Q_star_1, Q_star_0


def update_estimates(
    ps: np.ndarray, Y0_hat: np.ndarray, Y1_hat: np.ndarray, epsilon: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the initial outcome estimates using the fluctuation parameter.

    Parameters:
    -----------
    ps: array-like
        Propensity score
    Y0_hat: array-like
        Initial outcome prediction for control group
    Y1_hat: array-like
        Initial outcome prediction for treatment group
    epsilon: float
        Estimated fluctuation parameter

    Returns:
    --------
    tuple: (Q_star_1, Q_star_0)
        Updated outcome predictions for treatment and control groups
    """
    # Define clever covariates
    H1 = 1.0 / ps
    H0 = -1.0 / (1.0 - ps)

    # Update initial estimates with targeting step
    Q_star_1 = expit(logit(Y1_hat) + epsilon * H1)
    Q_star_0 = expit(logit(Y0_hat) + epsilon * H0)

    return Q_star_1, Q_star_0


def estimate_fluctuation_parameter(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, Yhat: np.ndarray
) -> float:
    """
    Estimate the fluctuation parameter epsilon using a logistic regression model.
    Returns the estimated epsilon.
    """
    # compute the clever covariate H
    H = A / ps - (1 - A) / (1 - ps)
    # Check for extreme values in clever covariate
    if np.any(np.abs(H) > 100):
        warnings.warn(
            "Extreme values detected in clever covariate H. "
            "This may indicate issues with propensity scores near 0 or 1.",
            RuntimeWarning,
        )
    # Use logit of the current outcome as offset
    offset = logit(Yhat)

    # Fit the model with offset
    model = GLM(Y, H, family=Binomial(), offset=offset).fit()
    return np.asarray(model.params)[0]
