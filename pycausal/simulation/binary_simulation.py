import numpy as np
import pandas as pd
from scipy.integrate import dblquad
from scipy.special import expit as logistic
from scipy.stats import norm


def simulate_binary_data(n: int, alpha: list, beta: list, seed=None) -> pd.DataFrame:
    """
    Simulate simple binary outcome data with two covariates and
    a binary treatment simulated from a logistic regression model.
    n: number of samples
    alpha: coefficients for the treatment model (intercept, X1, X2, X1*X2)
    beta: coefficients for the outcome model (intercept, A, X1, X2, X1*X2)
    """
    if seed is not None:
        # ise new generator with seed
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # Simulate covariates
    X = rng.normal(0, 1, (n, 2))
    X1 = X[:, 0]
    X2 = X[:, 1]
    # Simulate treatment
    logit_p = alpha[0] + alpha[1] * X1 + alpha[2] * X2 + alpha[3] * X1 * X2
    p = logistic(logit_p)
    A = rng.binomial(1, p)
    # Simulate outcome
    logit_q = beta[0] + beta[1] * A + beta[2] * X1 + beta[3] * X2 + beta[4] * X1 * X2
    q = logistic(logit_q)
    Y = rng.binomial(1, q)
    data = pd.DataFrame({"X1": X1, "X2": X2, "A": A, "Y": Y})

    return data


def compute_ATE_theoretical_from_data(data: pd.DataFrame, beta: list):
    """Compute the true average treatment effect (ATE) from the model coefficients, using the data."""
    E_Y1 = logistic(
        beta[0]
        + beta[1] * 1
        + beta[2] * data.X1
        + beta[3] * data.X2
        + beta[4] * data.X1 * data.X2
    ).mean()
    E_Y0 = logistic(
        beta[0]
        + beta[1] * 0
        + beta[2] * data.X1
        + beta[3] * data.X2
        + beta[4] * data.X1 * data.X2
    ).mean()
    return E_Y1 - E_Y0


def compute_ATE_theoretical_from_model(beta: list):
    """Compute the true average treatment effect (ATE) from the model coefficients, using the model."""
    beta_0, beta_1, beta_2, beta_3, beta_4 = beta

    # Function to integrate
    def integrand_1(x1, x2, beta0, beta1, beta2, beta3):
        return (
            logistic(beta0 + beta1 + beta2 * x1 + beta3 * x2)
            * norm.pdf(x1)
            * norm.pdf(x2)
        )

    def integrand_0(x1, x2, beta0, beta2, beta3):
        return logistic(beta0 + beta2 * x1 + beta3 * x2) * norm.pdf(x1) * norm.pdf(x2)

    # Double integration over normal distributions for X1 and X2 within finite bounds
    integration_bounds = (-10, 10)
    result_1, _ = dblquad(
        lambda x1, x2: integrand_1(x1, x2, beta_0, beta_1, beta_2, beta_3),
        integration_bounds[0],
        integration_bounds[1],
        lambda x: integration_bounds[0],
        lambda x: integration_bounds[1],
    )
    result_0, _ = dblquad(
        lambda x1, x2: integrand_0(x1, x2, beta_0, beta_2, beta_3),
        integration_bounds[0],
        integration_bounds[1],
        lambda x: integration_bounds[0],
        lambda x: integration_bounds[1],
    )

    # Theoretical ATE
    ATE_correct = result_1 - result_0

    return ATE_correct
