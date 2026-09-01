"""
Augmented Inverse Probability of Treatment Weighting (AIPW)
References:

ATE:
    Robins, James, Mariela Sued, Quanhong Lei-Gomez, and Andrea Rotnitzky.
    "Comment: Performance of double-robust estimators when 'inverse
    probability' weights are highly variable."
    Statistical Science 22.4 (2007): 544-559.
    Eq. (1)

ATT:
    Sant’Anna, Pedro HC, and Jun Zhao.
    "Doubly robust difference-in-differences estimators."
    Journal of econometrics 219.1 (2020): 101-122.
    Eq. 2.6
    code: https://github.com/pedrohcgs/DRDID/blob/master/R/drdid_imp_panel.R
"""

import warnings

import numpy as np

from CausalEstimate.estimators.functional.ipw import compute_ipw_weights
from CausalEstimate.utils.constants import EFFECT


def compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat) -> dict:
    """
    Augmented Inverse Probability of Treatment Weighting (AIPW) for ATE.
    A: treatment assignment, Y: outcome, ps: propensity score
    Y0_hat: P[Y|A=0], Y1_hat: P[Y|A=1]
    """
    W = compute_ipw_weights(A, ps, weight_type="ATE")
    w1, w0 = A * W, (1 - A) * W
    if (A == 1).sum() == 0:
        warnings.warn("No subjects in the treated group. mu_1 is NaN.", RuntimeWarning)
    if (A == 0).sum() == 0:
        warnings.warn("No subjects in the control group. mu_0 is NaN.", RuntimeWarning)
    mu_1 = (w1 * (Y - Y1_hat)).sum() / w1.sum() + Y1_hat.mean()
    mu_0 = (w0 * (Y - Y0_hat)).sum() / w0.sum() + Y0_hat.mean()
    ate = mu_1 - mu_0
    return {EFFECT: ate}


def compute_aipw_att(A, Y, ps, Y0_hat) -> dict:
    """
    Augmented Inverse Probability Weighting (AIPW) for ATT.
    A: treatment assignment (binary), Y: outcome, ps: propensity score
    Y0_hat: predicted outcome under control
    """
    S = compute_att_weights(A, ps)
    att = (S * (Y - Y0_hat)).sum()
    return {EFFECT: att}


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
