"""
Inverse Probability Weighting (IPW) estimators

References: 
ATE:
    Estimation of Average Treatment Effects Honors Thesis Peter Zhang
    https://lsa.umich.edu/content/dam/econ-assets/Econdocs/HonorsTheses/Estimation%20of%20Average%20Treatment%20Effects.pdf

    Austin, P.C., 2016. Variance estimation when using inverse probability of 
    treatment weighting (IPTW) with survival analysis. 
    Statistics in medicine, 35(30), pp.5642-5655.

ATT: 
    Reifeis et. al. (2022). 
    On variance of the treatment effect in the treated when estimated by 
    inverse probability weighting. 
    American Journal of Epidemiology, 191(6), 1092-1097.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9271225/
"""


def compute_ipw_risk_ratio(A, Y, ps):
    """
    Relative Risk
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu_1, mu_0 = compute_mean_potential_outcomes(A, Y, ps)
    return mu_1 / mu_0


def compute_ipw_ate(A, Y, ps):
    """
    Average Treatment Effect
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu1, mu0 = compute_mean_potential_outcomes(A, Y, ps)
    return mu1 - mu0


def compute_mean_potential_outcomes(A, Y, ps):
    """
    Compute E[Y|A=1] and E[Y|A=0] for Y=0/1
    """
    mu_1 = (A * Y / ps).mean()
    mu_0 = ((1 - A) * Y / (1 - ps)).mean()
    return mu_1, mu_0


def compute_ipw_ate_stabilized(A, Y, ps):
    """
    Given by Austin (2016)
    Average Treatment Effect with stabilized weights.
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    W = compute_stabilized_ate_weights(A, ps)
    Y1_weighed = (W * A * Y).mean()
    Y0_weighed = (W * (1 - A) * Y).mean()
    return Y1_weighed - Y0_weighed


def compute_ipw_att(A, Y, ps):
    """
    Average Treatment Effect on the Treated with stabilized weights.
    Reifeis et. al. (2022).
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu_1, mu_0 = compute_mean_potential_outcomes_treated(A, Y, ps)
    return mu_1 - mu_0


def compute_ipw_risk_ratio_treated(A, Y, ps):
    """
    Relative Risk of the Treated with stabilized weights. Reifeis et. al. (2022)
    A: treatment assignment, Y: outcome, ps: propensity score
    """
    mu_1, mu_0 = compute_mean_potential_outcomes_treated(A, Y, ps)
    return mu_1 / mu_0


def compute_mean_potential_outcomes_treated(A, Y, ps):
    """
    Compute E[Y|A=1] for Y=0/1
    """
    W = compute_stabilized_att_weights(A, ps)
    mu_1 = (W * A * Y).sum() / (W * A).sum()
    mu_0 = (W * (1 - A) * Y).sum() / (W * (1 - A)).sum()
    return mu_1, mu_0


def compute_stabilized_ate_weights(A, ps):
    """
    Compute the (stabilized) weights for the ATE estimator
    Austin (2016)
    """
    weight_treated = A.mean() * A / ps
    weight_control = (1 - A).mean() * (1 - A) / (1 - ps)
    return weight_treated + weight_control


def compute_stabilized_att_weights(A, ps):
    """
    Compute the (stabilized) weights for the ATT estimator
    As given in the web appendix of Reifeis et. al. (2022)
    """
    h = ps / (1 - ps)
    return A + (1 - A) * h
