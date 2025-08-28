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
import statsmodels.tools.sm_exceptions as sm_exceptions
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
        # Check for practically zero denominator (not just exact zero)
        if np.isclose(initial_effect_0, 0, atol=1e-8):
            import warnings

            warnings.warn(
                "Initial effect for untreated group is 0 or very close to 0, risk ratio undefined",
                RuntimeWarning,
            )
            initial_effect = np.inf
        else:
            # Also check if the resulting ratio would be unrealistically large
            ratio = initial_effect_1 / initial_effect_0
            if np.abs(ratio) > 1e5:  # Threshold for "unrealistically large"
                warnings.warn(
                    f"Risk ratio is unrealistically large ({ratio:.2e}), setting to inf",
                    RuntimeWarning,
                )
                initial_effect = np.inf
            else:
                initial_effect = ratio
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
    Robustly estimates the fluctuation parameter epsilon for the TMLE targeting step.

    This function attempts to fit a standard iterative Maximum Likelihood Estimate (MLE)
    for epsilon. It includes safety checks for common failure modes of the logistic
    regression, such as perfect separation, convergence failure, or numerically
    unstable (i.e., extremely large) estimates.

    If any of these issues are detected, it issues a warning and falls back to the
    computationally stable `_compute_epsilon_one_step` function.

    Args:
        H: The clever covariate array, H(A,X).
        Y: The binary outcome array.
        Yhat: The initial predictions for the outcome, Q(A,X).

    Returns:
        The estimated epsilon, choosing the most stable and reliable result.
    """
    # Clip Yhat to avoid ±inf in logit transformation
    eps = 1e-6
    Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
    offset = logit(Yhat_clipped)

    epsilon_mle = np.nan
    converged = False

    # --- 1. Attempt to fit the standard MLE for epsilon ---
    # We catch both explicit errors (like perfect separation) and implicit
    # convergence warnings from the statsmodels GLM fitter.
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", sm_exceptions.ConvergenceWarning)

            # Reshape H for statsmodels (it expects a 2D array for the predictor)
            H_2d = H.reshape(-1, 1)
            model = GLM(Y, H_2d, family=Binomial(), offset=offset).fit()

            # Check if a ConvergenceWarning was issued during the fit
            if not any(
                issubclass(warn.category, sm_exceptions.ConvergenceWarning)
                for warn in w
            ):
                converged = True

            epsilon_mle = np.asarray(model.params)[0]

    except (np.linalg.LinAlgError, sm_exceptions.PerfectSeparationError):
        # Catch explicit errors where the model cannot be fit at all.
        # epsilon_mle remains NaN and converged remains False.
        pass

    # --- 2. Check if the MLE result is stable and reliable ---
    # The heuristic checks if epsilon is non-finite, if the GLM failed to converge,
    # or if the magnitude of the logit-scale update is extreme (>15) for 95% of the data.
    p95_H = np.quantile(np.abs(H), 0.95)
    is_unstable = (
        not converged
        or not np.isfinite(epsilon_mle)
        or (np.abs(epsilon_mle) * p95_H > 15)
    )

    # --- 3. Return the MLE or fall back to the one-step estimate ---
    if is_unstable:
        warnings.warn(
            "MLE for epsilon failed to converge or was numerically unstable. "
            "Falling back to the one-step estimate.",
            RuntimeWarning,
        )
        return _compute_epsilon_one_step(H, Y, Yhat_clipped)
    else:
        return epsilon_mle


def _compute_epsilon_one_step(H: np.ndarray, Y: np.ndarray, Yhat: np.ndarray) -> float:
    """
    Computes a stable, non-iterative one-step estimate for epsilon.

    This serves as a robust fallback if the iterative MLE fails or is unstable.
    The one-step estimate is the first step of the Newton-Raphson algorithm for
    solving the score equation of the logistic fluctuation model, starting from epsilon = 0.

    The formula is:
        epsilon_1-step = Score(0) / Information(0)

    where Score(0) is the gradient and Information(0) is the observed information
    (negative second derivative) of the log-likelihood, both evaluated at epsilon = 0.

    Args:
        H: The clever covariate array, H(A,X).
        Y: The binary outcome array.
        Yhat: The initial predictions for the outcome, Q(A,X). Should be pre-clipped.

    Returns:
        The one-step epsilon estimate.
    """
    # The 'score' is the gradient of the log-likelihood at epsilon=0.
    # d(logL)/d(epsilon)|_eps=0 = sum(H * (Y - P(Y=1|A,X,eps=0))) = sum(H * (Y - Yhat))
    score = np.sum(H * (Y - Yhat))

    # The 'information' is the negative of the second derivative (Hessian).
    # -d^2(logL)/d(epsilon)^2|_eps=0 = sum(H^2 * Var(Y|A,X)) = sum(H^2 * Yhat * (1-Yhat))
    information = np.sum(H**2 * Yhat * (1 - Yhat))

    # Guard against non-finite score or information
    if not np.isfinite(score) or not np.isfinite(information) or information == 0:
        warnings.warn(
            "Non-finite score or information in one-step epsilon estimation. "
            "Returning epsilon = np.nan.",
            RuntimeWarning,
        )
        return np.nan

    return score / information
