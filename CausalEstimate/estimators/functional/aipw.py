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


def compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat):
    """
    Augmented Inverse Probability of Treatment Weighting (AIPW) for ATE.
    A: treatment assignment, Y: outcome, ps: propensity score
    Y0_hat: P[Y|A=0], Y1_hat: P[Y|A=1]
    """
    ate_ipw = compute_ipw_ate(A, Y, ps)
    adjustment_factor = (A - ps) / (ps * (1 - ps))
    ate = ate_ipw - adjustment_factor * ((1 - ps) * Y1_hat + ps * Y0_hat)
    return ate.mean()
