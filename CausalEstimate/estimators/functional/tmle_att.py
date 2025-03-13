"""
The implementation is largely based on the following reference:
Van der Laan MJ, Rose S. Targeted learning: causal inference for observational and experimental data. Springer; New York: 2011. Specifically, Chapter 8 for the ATT TMLE.
But slightly modified for simpler implementation, following advice from: https://stats.stackexchange.com/questions/520472/can-targeted-maximum-likelihood-estimation-find-the-average-treatment-effect-on/534018#534018
"""

import warnings
from typing import Tuple

import numpy as np
from scipy.special import expit, logit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated


def compute_estimates_att(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute updated outcome estimates for ATT using a one-step TMLE targeting step.

    For ATT, the clever covariate is defined as:
      H(A,W) = 1{A=1}/P(A=1) - 1{A=0}*ps/(P(A=1)*(1-ps)),
    where P(A=1) is the empirical probability of treatment.

    Parameters:
    -----------
    A: array-like
         Treatment assignment (0 or 1)
    Y: array-like
         Binary outcome
    ps: array-like
         Propensity score P(A=1|X)
    Y0_hat: array-like
         Initial outcome prediction for control group P(Y=1|A=0,X)
    Y1_hat: array-like
         Initial outcome prediction for treatment group P(Y=1|A=1,X)
    Yhat: array-like
         Combined outcome prediction = A*Y1_hat + (1-A)*Y0_hat

    Returns:
    --------
    tuple: (Q_star_1, Q_star_0)
         Updated outcome predictions for treated and control groups.
    """
    # Empirical probability of treatment
    p_treated = np.mean(A == 1)

    # Estimate the fluctuation parameter epsilon using a logistic regression:
    epsilon = estimate_fluctuation_parameter_att(A, Y, ps, Yhat)

    # Update outcome predictions:
    # For treated group: H = 1/p_treated, so update is: logit(Y1_hat) + epsilon*(1/p_treated)
    # For control group: H = -ps/(p_treated*(1-ps)), so update is: logit(Y0_hat) - epsilon*(ps/(p_treated*(1-ps)))
    Q_star_1 = expit(logit(Y1_hat) + epsilon * (1.0 / p_treated))
    Q_star_0 = expit(logit(Y0_hat) - epsilon * (ps / (p_treated * (1 - ps))))

    return Q_star_1, Q_star_0


def estimate_fluctuation_parameter_att(
    A: np.ndarray, Y: np.ndarray, ps: np.ndarray, Yhat: np.ndarray
) -> float:
    """
    Estimate the fluctuation parameter epsilon for the ATT TMLE via logistic regression.

    The model is:
         logit(Q*(A,W)) = logit(Yhat) + epsilon * H,
    and is fitted with no intercept.

    Parameters:
    -----------
    A: array-like
         Treatment assignment (0 or 1)
    Y: array-like
         Binary outcome
    ps: array-like
         Propensity score P(A=1|X)
    Yhat: array-like
         Combined outcome prediction (A*Y1_hat + (1-A)*Y0_hat)

    Returns:
    --------
    float: Estimated fluctuation parameter epsilon.
    """
    p_treated = np.mean(A == 1)
    # Define the clever covariate H for each observation:
    # For treated: 1/p_treated, for controls: -ps/(p_treated*(1-ps))
    H = (A / p_treated) - ((1 - A) * ps / (p_treated * (1 - ps)))
    # Check for extreme values in H
    if np.any(np.abs(H) > 100):
        warnings.warn(
            "Extreme values detected in clever covariate H. "
            "This may indicate issues with propensity scores near 0 or 1.",
            RuntimeWarning,
        )
    # Reshape H for statsmodels (needs a 2D array)
    H_2d = H.reshape(-1, 1)
    offset = logit(Yhat)

    # Fit logistic regression with no intercept (i.e. model: Y ~ -1 + H, with offset)
    model = GLM(Y, H_2d, family=Binomial(), offset=offset)
    results = model.fit()

    epsilon = results.params[0]
    return epsilon


def compute_tmle_att(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y0_hat: np.ndarray,
    Y1_hat: np.ndarray,
    Yhat: np.ndarray,
) -> dict:
    """
    Estimate the Average Treatment Effect on the Treated (ATT) using TMLE.

    The TMLE ATT estimator is given by:
      Ψ* = (1/n) * sum [ ps(W_i)/P(A=1) * ( Q*_n(1,W_i) - Q*_n(0,W_i) ) ],
    where Q*_n(1,W) and Q*_n(0,W) are the updated outcome predictions.

    Parameters:
    -----------
    A: array-like
         Treatment assignment (0 or 1)
    Y: array-like
         Binary outcome
    ps: array-like
         Propensity score P(A=1|X)
    Y0_hat: array-like
         Initial outcome prediction for controls
    Y1_hat: array-like
         Initial outcome prediction for treated
    Yhat: array-like
         Combined outcome prediction = A*Y1_hat + (1-A)*Y0_hat

    Returns:
    --------
    float: ATT estimate.
    """
    Q_star_1, Q_star_0 = compute_estimates_att(A, Y, ps, Y0_hat, Y1_hat, Yhat)
    p_treated = np.mean(A == 1)

    # Compute plugin ATT estimator: average of (Q*_1 - Q*_0) over treated units reweighted by ps/P(A=1)
    psi = np.mean((ps / p_treated) * (Q_star_1 - Q_star_0))
    return {EFFECT: psi, EFFECT_treated: Q_star_1, EFFECT_untreated: Q_star_0}
