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

ATC:
    The ATT estimator with the roles of the two arms swapped.
"""

import warnings

import numpy as np

from CausalEstimate.estimators.functional.utils import (
    att_result_as_atc,
    compute_ipw_weights,
    safe_ratio,
)
from CausalEstimate.estimators.functional.variance import compute_ci_aipw
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
    W, mu_1, mu_0 = _aipw_marginal_means(A, Y, ps, Y0_hat, Y1_hat, clip_percentile, eps)
    ate = mu_1 - mu_0
    ci_results = compute_ci_aipw(
        effect_type="ATE",
        psi=ate,
        Y=Y,
        A=A,
        W=W,
        Q_1=Y1_hat,
        Q_0=Y0_hat,
        mu_1=mu_1,
        mu_0=mu_0,
        eps=eps,
    )
    return {EFFECT: ate, EFFECT_treated: mu_1, EFFECT_untreated: mu_0, **ci_results}


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
    W, mu_1, mu_0 = _aipw_treated_means(
        A, Y, ps, Y0_hat, clip_percentile, eps, label="ATT"
    )
    att = mu_1 - mu_0
    ci_results = compute_ci_aipw(
        effect_type="ATT",
        psi=att,
        Y=Y,
        A=A,
        W=W,
        Q_1=None,  # unused for ATT: mu_1 is the observed treated mean
        Q_0=Y0_hat,
        mu_1=mu_1,
        mu_0=mu_0,
        eps=eps,
    )
    return {EFFECT: att, EFFECT_treated: mu_1, EFFECT_untreated: mu_0, **ci_results}


def compute_aipw_atc(
    A, Y, ps, Y1_hat, clip_percentile: float = 1, eps: float = 1e-9
) -> dict:
    """
    Augmented Inverse Probability Weighting (AIPW) for the ATC, as the ATT
    with the arms swapped (see `att_result_as_atc`). Only Y1_hat is needed:
    it plays the role Y0_hat plays in the ATT.
    """
    return att_result_as_atc(
        compute_aipw_att(
            1 - A, Y, 1 - ps, Y1_hat, clip_percentile=clip_percentile, eps=eps
        )
    )


def compute_aipw_rr(
    A, Y, ps, Y0_hat, Y1_hat, clip_percentile: float = 1, eps: float = 1e-9
) -> dict:
    """
    Augmented Inverse Probability Weighting (AIPW) for the Risk Ratio.

    The point estimate is the ratio of the same doubly robust potential-outcome
    means used by compute_aipw_ate, so RR and ATE stay consistent with one
    another.

    The ratio goes through `safe_ratio`, as the TMLE RR does, so an empty arm
    gives NaN and a (near-)zero mu_0 gives inf in both estimators.

    The CI is the delta method on the log scale: the arm-level influence curves
    are combined as IC_mu1/mu_1 - IC_mu0/mu_0, matching how the TMLE RR is
    handled. STD_ERR is therefore on the log scale, and CI95 is its
    exponentiation.
    """
    W, mu_1, mu_0 = _aipw_marginal_means(A, Y, ps, Y0_hat, Y1_hat, clip_percentile, eps)

    _warn_if_not_risks(mu_1, mu_0, label="RR")
    rr = safe_ratio(mu_1, mu_0, label="Risk ratio")

    ci_results = compute_ci_aipw(
        effect_type="RR",
        psi=rr,
        Y=Y,
        A=A,
        W=W,
        Q_1=Y1_hat,
        Q_0=Y0_hat,
        mu_1=mu_1,
        mu_0=mu_0,
        eps=eps,
    )
    return {EFFECT: rr, EFFECT_treated: mu_1, EFFECT_untreated: mu_0, **ci_results}


def compute_aipw_rrt(
    A, Y, ps, Y0_hat, clip_percentile: float = 1, eps: float = 1e-9
) -> dict:
    """
    Augmented Inverse Probability Weighting (AIPW) for the Risk Ratio in the
    Treated.

    As with compute_aipw_att, only Y0_hat is needed: mu_1 is the observed
    treated mean, and Y1_hat would cancel from the influence curve anyway.

    The influence curve for mu_0 centres its plug-in term on A/P(A=1) rather
    than on 1, because mu_0 is a mean over the treated subpopulation. STD_ERR
    is on the log scale, and CI95 is its exponentiation. As for the RR, the
    ratio goes through `safe_ratio`.
    """
    W, mu_1, mu_0 = _aipw_treated_means(
        A, Y, ps, Y0_hat, clip_percentile, eps, label="RRT"
    )

    _warn_if_not_risks(mu_1, mu_0, label="RRT")
    rrt = safe_ratio(mu_1, mu_0, label="Risk ratio in the treated")

    ci_results = compute_ci_aipw(
        effect_type="RRT",
        psi=rrt,
        Y=Y,
        A=A,
        W=W,
        Q_1=None,  # unused for RRT: mu_1 is the observed treated mean
        Q_0=Y0_hat,
        mu_1=mu_1,
        mu_0=mu_0,
        eps=eps,
    )
    return {EFFECT: rrt, EFFECT_treated: mu_1, EFFECT_untreated: mu_0, **ci_results}


def _warn_if_not_risks(mu_1, mu_0, label):
    """
    AIPW arm means are not bounded to [0, 1]: with a rare outcome and extreme
    weights the augmentation can push one outside it, where the ratio is not a
    risk ratio (and at or below zero its log-scale CI is NaN). Say so rather
    than report it silently.
    """
    for name, mu in (("mu_1", mu_1), ("mu_0", mu_0)):
        if np.isfinite(mu) and not (0 < mu <= 1):
            warnings.warn(
                f"AIPW arm mean {name} = {mu:.3g} is outside (0, 1], so the "
                f"{label} is not a valid risk ratio. This usually means "
                "extreme weights; consider clip_percentile, trimming, or TMLE, "
                "whose arm means stay in [0, 1].",
                RuntimeWarning,
            )


def _aipw_marginal_means(A, Y, ps, Y0_hat, Y1_hat, clip_percentile, eps):
    """
    ATE weights and the doubly robust marginal means E[Y(1)] and E[Y(0)],
    shared by the ATE and the RR so the two cannot disagree.
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
    return W, mu_1, mu_0


def _aipw_treated_means(A, Y, ps, Y0_hat, clip_percentile, eps, label):
    """
    ATT weights, the observed treated mean mu_1 and the doubly robust
    counterfactual mean mu_0 = E[Y(0) | A=1], shared by the ATT and the RRT.
    """
    if (A == 1).sum() == 0:
        warnings.warn(
            f"No subjects in the treated group. {label} is NaN.", RuntimeWarning
        )
    if (A == 0).sum() == 0:
        warnings.warn(
            f"No subjects in the control group. {label} is NaN.", RuntimeWarning
        )
    W = compute_ipw_weights(
        A, ps, weight_type="ATT", clip_percentile=clip_percentile, eps=eps
    )
    w0 = (1 - A) * W
    treated = A == 1
    mu_1 = Y[treated].mean()
    mu_0 = Y0_hat[treated].mean() + (w0 * (Y - Y0_hat)).sum() / w0.sum()
    return W, mu_1, mu_0
