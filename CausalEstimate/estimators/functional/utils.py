import numpy as np
import warnings

from CausalEstimate.utils.constants import (
    INITIAL_EFFECT,
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
)
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from scipy.special import logit


def compute_clever_covariate_ate(
    A: np.ndarray,
    ps: np.ndarray,
    stabilized: bool = False,
) -> np.ndarray:
    """
    Compute the clever covariate H for ATE TMLE.

    Parameters:
    -----------
    A: np.ndarray
        Treatment assignment (0 or 1)
    ps: np.ndarray
        Propensity scores
    stabilized: bool, optional
        Whether to use stabilized weights. Default is False.

    Returns:
    --------
    np.ndarray: The clever covariate H
    """
    if stabilized:
        pi = A.mean()
        # Stabilized clever covariate
        H = A * pi / ps - (1 - A) * (1 - pi) / (1 - ps)
    else:
        # Unstabilized clever covariate
        H = A / ps - (1 - A) / (1 - ps)

    _validate_clever_covariate(H, "ATE")
    return H


def compute_clever_covariate_att(
    A: np.ndarray,
    ps: np.ndarray,
    stabilized: bool = False,
) -> np.ndarray:
    """
    Compute the clever covariate H for ATT TMLE.

    Parameters:
    -----------
    A: np.ndarray
        Treatment assignment (0 or 1)
    ps: np.ndarray
        Propensity scores
    stabilized: bool, optional
        Whether to use stabilized weights. Default is False.

    Returns:
    --------
    np.ndarray: The clever covariate H
    """
    p_treated = np.mean(A == 1)
    if p_treated == 0:
        warnings.warn(
            "No treated subjects found, returning zeros for H.", RuntimeWarning
        )
        return np.zeros_like(A, dtype=float)

    # Component for treated individuals
    H_treated = A / p_treated

    # Component for control individuals
    if stabilized:
        # Stabilized clever covariate for controls
        H_control = (1 - A) * ps * (1 - p_treated) / (p_treated * (1 - ps))
    else:
        # Unstabilized clever covariate for controls
        H_control = (1 - A) * ps / (p_treated * (1 - ps))

    H = H_treated - H_control
    _validate_clever_covariate(H, "ATT")
    return H


def _validate_clever_covariate(H: np.ndarray, estimand: str) -> None:
    """
    Validate the clever covariate for potential numerical issues.

    Parameters:
    -----------
    H: np.ndarray
        The clever covariate
    estimand: str
        The estimand type ("ATE" or "ATT") for informative warnings
    """
    if np.any(np.abs(H) > 100):
        warnings.warn(
            f"Extremely large values > 100 detected in clever covariate H for {estimand}. "
            "This may indicate issues with propensity scores near 0 or 1.",
            RuntimeWarning,
        )
    if np.any(np.abs(H) < 1e-6):
        warnings.warn(
            f"Extremely small values < 1e-6 detected in clever covariate H for {estimand}. "
            "This may indicate issues with propensity scores near 0 or 1.",
            RuntimeWarning,
        )


def compute_initial_effect(
    Y1_hat: np.ndarray,
    Y0_hat: np.ndarray,
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    rr: bool = False,
) -> dict:
    """
    Compute the initial effect and adjustments.

    Parameters:
    -----------
    Y1_hat: array-like
        Initial outcome prediction for treatment group
    Y0_hat: array-like
        Initial outcome prediction for control group
    Q_star_1: array-like
        Updated outcome predictions for treatment group
    Q_star_0: array-like
        Updated outcome predictions for control group
    rr: bool, optional
        If True, compute the risk ratio. If False, compute the average treatment effect.

    Returns:
    --------
    dict: A dictionary containing the initial effect and adjustments.
    """
    initial_effect_1 = Y1_hat.mean()
    initial_effect_0 = Y0_hat.mean()

    if rr:
        if initial_effect_0 == 0:
            import warnings

            warnings.warn(
                "Initial effect for untreated group is 0, risk ratio undefined",
                RuntimeWarning,
            )
            initial_effect = np.inf
        else:
            initial_effect = initial_effect_1 / initial_effect_0
    else:
        initial_effect = initial_effect_1 - initial_effect_0

    adjustment_1 = (Q_star_1 - Y1_hat).mean()
    adjustment_0 = (Q_star_0 - Y0_hat).mean()
    return {
        INITIAL_EFFECT: initial_effect,
        INITIAL_EFFECT_treated: initial_effect_1,
        INITIAL_EFFECT_untreated: initial_effect_0,
        ADJUSTMENT_treated: adjustment_1,
        ADJUSTMENT_untreated: adjustment_0,
    }


def estimate_fluctuation_parameter(
    H: np.ndarray,
    Y: np.ndarray,
    Yhat: np.ndarray,
) -> float:
    """
    Estimate the fluctuation parameter epsilon using a logistic regression model.
    """

    offset = logit(Yhat)
    model = GLM(Y, H, family=Binomial(), offset=offset).fit()

    # model.params is a one-element array containing epsilon
    return np.asarray(model.params)[0]
