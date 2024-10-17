"""
Augmented Inverse Probability of Treatment Weighting (AIPW)
References:

ATE:
        Glynn, Adam N., and Kevin M. Quinn.
        "An introduction to the augmented inverse propensity weighted estimator." 
        Political analysis 18.1 (2010): 36-56.
        note: This also provides a variance estimator for the AIPW estimator.


"""

from CausalEstimate.estimators.functional.ipw import compute_ipw_ate
import numpy as np

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


def compute_stabilized_att_weights(A, ps)->np.ndarray:
    """
    Compute the stabilized weights for the ATT estimator.
    """
    h = ps / (1 - ps)
    return A + (1 - A) * h


def compute_aipw_att(A, Y, ps, Y0_hat, Y1_hat)->float:
    """
    Augmented Inverse Probability Weighting (AIPW) for ATT.
    A: treatment assignment (binary), Y: outcome, ps: propensity score
    Y0_hat: predicted outcome under control, Y1_hat: predicted outcome under treatment
    """
    # Compute the stabilized weights for ATT
    W = compute_stabilized_att_weights(A, ps)

    # Numerator for the IPW ATT estimator
    ipw_numer = (W * A * Y).sum() - (W * (1 - A) * Y).sum()
    # Denominator for the IPW ATT estimator
    ipw_denom = (W * A).sum()
    # IPW ATT estimate
    ipw_att = ipw_numer / ipw_denom

    # Augmentation term
    augmentation = ((W * (1 - A) * (Y0_hat - Y1_hat)).sum()) / (W * A).sum()

    # Predicted means for treated units
    mu1_hat = Y1_hat[A == 1].mean()
    mu0_hat = Y0_hat[A == 1].mean()

    # Compute the AIPW ATT
    att = ipw_att + augmentation + (mu1_hat - mu0_hat)

    return att


def compute_adjustment_factor(A, ps)->np.ndarray:
    """Compute the adjustment factor for the AIPW estimator."""
    return (A - ps) / (ps * (1 - ps))
