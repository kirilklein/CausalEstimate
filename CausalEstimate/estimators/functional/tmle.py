from scipy.special import expit, logit
from statsmodels.api import add_constant
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM


def compute_tmle_ate(A, Y, ps, Y0_hat, Y1_hat, Yhat):
    """
    Estimate the average treatment effect using the targeted maximum likelihood estimation (TMLE) method.
    A: treatment assignment, Y: outcome, ps: propensity score,
    Y0_hat: P[Y|A=0], Y1_hat: P[Y|A=1], Yhat: P[Y]
    """
    epsilon = estimate_fluctuation_parameter(A, Y, ps, Yhat)
    return update_ate_estimate(ps, Y0_hat, Y1_hat, epsilon)


def update_ate_estimate(ps, Y0_hat, Y1_hat, epsilon) -> tuple:
    """Update the Q_star values using the fluctuation parameter epsilon."""
    H_1 = 1 / ps
    Q_star_1 = expit(logit(Y1_hat) + epsilon * H_1)

    H_0 = 1 / (1 - ps)
    Q_star_0 = expit(logit(Y0_hat) - epsilon * H_0)

    return (Q_star_1 - Q_star_0).mean()


def estimate_fluctuation_parameter(A, Y, ps, Yhat) -> float:
    """
    Estimate the fluctuation parameter epsilon using a logistic regression model.
    Returns the estimated epsilon.
    """
    # compute the clever covariate H
    H = A / ps - (1 - A) / (1 - ps)

    # Use logit of the current outcome as offset
    offset = logit(Yhat)

    # Fit the model with offset
    model = GLM(Y, add_constant(H), family=Binomial(), offset=offset).fit()
    return model.params[0]
