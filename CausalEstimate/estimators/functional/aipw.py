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

from CausalEstimate.estimators.functional.ipw import compute_ipw_weights
from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated


def compute_aipw_ate(
    A, Y, ps, Y0_hat, Y1_hat, clip_percentile: float = 1, eps: float = 1e-9
) -> dict:
    """
    Augmented Inverse Probability of Treatment Weighting (AIPW) for ATE.
    A: treatment assignment, Y: outcome, ps: propensity score
    Y0_hat: P[Y|A=0], Y1_hat: P[Y|A=1]
    clip_percentile: upper percentile at which to clip weights (1 = no clipping)
    eps: small constant added to denominators for numerical stability

    Returns the effect together with the potential-outcome means mu_1 and mu_0,
    so MultiEstimator can summarise effect_1 / effect_0 as it does for the
    other estimators.
    """
    W = compute_ipw_weights(
        A, ps, weight_type="ATE", clip_percentile=clip_percentile, eps=eps
    )
    w1, w0 = A * W, (1 - A) * W
    if (A == 1).sum() == 0:
        warnings.warn("No subjects in the treated group. mu_1 is NaN.", RuntimeWarning)
    if (A == 0).sum() == 0:
        warnings.warn("No subjects in the control group. mu_0 is NaN.", RuntimeWarning)
    mu_1 = (w1 * (Y - Y1_hat)).sum() / w1.sum() + Y1_hat.mean()
    mu_0 = (w0 * (Y - Y0_hat)).sum() / w0.sum() + Y0_hat.mean()
    ate = mu_1 - mu_0
    return {EFFECT: ate, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}


def compute_aipw_att(
    A, Y, ps, Y0_hat, clip_percentile: float = 1, eps: float = 1e-9
) -> dict:
    """
    Augmented Inverse Probability Weighting (AIPW) for ATT.
    A: treatment assignment (binary), Y: outcome, ps: propensity score
    Y0_hat: predicted outcome under control
    clip_percentile: upper percentile at which to clip weights (1 = no clipping)
    eps: small constant added to denominators for numerical stability

    Returns the effect together with mu_1 (observed treated mean) and mu_0
    (counterfactual untreated mean for the treated).
    """
    if (A == 1).sum() == 0:
        warnings.warn("No subjects in the treated group. ATT is NaN.", RuntimeWarning)
    if (A == 0).sum() == 0:
        warnings.warn("No subjects in the control group. ATT is NaN.", RuntimeWarning)
    W = compute_ipw_weights(
        A, ps, weight_type="ATT", clip_percentile=clip_percentile, eps=eps
    )
    w0 = (1 - A) * W
    treated = A == 1
    mu_1 = Y[treated].mean()
    mu_0 = Y0_hat[treated].mean() + (w0 * (Y - Y0_hat)).sum() / w0.sum()
    att = mu_1 - mu_0
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0}
