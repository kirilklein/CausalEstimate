"""
Augmented Inverse Probability of Treatment Weighting (AIPW)
References:

ATE:
    Glynn, Adam N., and Kevin M. Quinn.
    "An introduction to the augmented inverse propensity weighted estimator." 
    Political analysis 18.1 (2010): 36-56.
    note: This also provides a variance estimator for the AIPW estimator.

ATT:
    Sant’Anna, Pedro HC, and Jun Zhao. 
    "Doubly robust difference-in-differences estimators." 
    Journal of econometrics 219.1 (2020): 101-122.
    Eq. 2.6
    code: https://github.com/pedrohcgs/DRDID/blob/master/R/drdid_imp_panel.R
"""

from typing import Tuple

import numpy as np

from CausalEstimate.estimators.functional.ipw import compute_ipw_ate


def compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat):
    """
    Augmented Inverse Probability of Treatment Weighting (AIPW) for ATE.
    A: treatment assignment, Y: outcome, ps: propensity score
    Y0_hat: P[Y|A=0], Y1_hat: P[Y|A=1]
    """
    ate_ipw = compute_ipw_ate(A, Y, ps)
    adjustment_factor = compute_adjustment_factor(A, ps)
    ate = ate_ipw - adjustment_factor * ((1 - ps) * Y1_hat + ps * Y0_hat)
    return ate.mean()


def compute_aipw_att(A, Y, ps, Y0_hat) -> float:
    """
    Augmented Inverse Probability Weighting (AIPW) for ATT.
    A: treatment assignment (binary), Y: outcome, ps: propensity score
    Y0_hat: predicted outcome under control
    """
    S = compute_att_weights(A, ps)
    return (S * (Y - Y0_hat)).sum()


def compute_adjustment_factor(A, ps) -> np.ndarray:
    """Compute the adjustment factor for the AIPW estimator."""
    return (A - ps) / (ps * (1 - ps))


def compute_att_weights(A, ps) -> np.ndarray:
    """
    Compute the weights for the ATT estimator.
    """
    w = ps / (1 - ps)
    n_treated = (A == 1).sum()
    scaling_treated = 1 / n_treated
    control_factor = (1 - A) * w
    scaling_control = 1 / control_factor.sum()
    return A * scaling_treated - control_factor * scaling_control


def compute_ipw_att_estimator(W: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
    """Compute the IPW ATT estimate."""
    numerator = (W * A * Y).sum() - (W * (1 - A) * Y).sum()
    denominator = (W * A).sum()
    return numerator / denominator


def compute_augmentation_term(
    W: np.ndarray, A: np.ndarray, Y0_hat: np.ndarray, Y1_hat: np.ndarray
) -> float:
    """Compute the augmentation term."""
    numerator = (W * (1 - A) * (Y0_hat - Y1_hat)).sum()
    denominator = (W * A).sum()
    return numerator / denominator


def compute_predicted_means_treated(
    Y0_hat: np.ndarray, Y1_hat: np.ndarray, A: np.ndarray
) -> Tuple[float, float]:
    """Compute predicted means for treated units."""
    treated_indices = A == 1
    mu1_hat = Y1_hat[treated_indices].mean()
    mu0_hat = Y0_hat[treated_indices].mean()
    return mu1_hat, mu0_hat
