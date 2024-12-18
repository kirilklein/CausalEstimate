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

    # alpha can be a list of length 3. Extend to length 6 with 0s
    if len(alpha) < 6:
        alpha = np.pad(alpha, (0, 6 - len(alpha)), mode="constant")
    # Simulate treatment
    logit_p = (
        alpha[0]
        + alpha[1] * X1
        + alpha[2] * X2
        + alpha[3] * X1 * X2
        + alpha[4] * X1**2
        + alpha[5] * X2**2
    )

    p = logistic(logit_p)
    A = rng.binomial(1, p)
    if len(beta) < 8:
        beta = np.pad(beta, (0, 8 - len(beta)), mode="constant")
    # Simulate outcome
    logit_q = (
        beta[0]
        + beta[1] * A
        + beta[2] * X1
        + beta[3] * X2
        + beta[4] * X1 * X2
        + beta[5] * X1**2
        + beta[6] * X2**2
        + beta[7] * A**2
    )
    q = logistic(logit_q)
    Y = rng.binomial(1, q)

    logit_q_cf = (
        beta[0] + beta[1] * (1 - A) + beta[2] * X1 + beta[3] * X2 + beta[4] * X1 * X2
    )
    q_cf = logistic(logit_q_cf)
    Y_cf = rng.binomial(1, q_cf)

    data = pd.DataFrame({"X1": X1, "X2": X2, "A": A, "Y": Y, "Y_cf": Y_cf})

    return data


def compute_expected_outcome(data: pd.DataFrame, beta: list, treatment: int):
    """Compute the expected outcome for a given treatment."""
    # extend beta to length 8 with 0s
    if len(beta) < 8:
        beta = np.pad(beta, (0, 8 - len(beta)), mode="constant")
    return logistic(
        beta[0]
        + beta[1] * treatment
        + beta[2] * data.X1
        + beta[3] * data.X2
        + beta[4] * data.X1 * data.X2
        + beta[5] * data.X1**2
        + beta[6] * data.X2**2
        + beta[7] * treatment**2
    ).mean()


def compute_ATE_theoretical_from_data(data: pd.DataFrame, beta: list):
    """Compute the true average treatment effect (ATE) from the model coefficients, using the data."""
    E_Y1 = compute_expected_outcome(data, beta, 1)
    E_Y0 = compute_expected_outcome(data, beta, 0)
    return E_Y1 - E_Y0


def compute_ATT_theoretical_from_data(data: pd.DataFrame, beta: list):
    """Compute the true average treatment effect on the treated (ATT) from the model coefficients, using the data."""
    treated_data = data[data.A == 1]
    E_Y1 = compute_expected_outcome(treated_data, beta, 1)
    E_Y0 = compute_expected_outcome(treated_data, beta, 0)
    return E_Y1 - E_Y0


def compute_ATE_theoretical_from_model(beta: list):
    """Compute the true average treatment effect (ATE) from the model coefficients, using the model."""
    # extend beta to length 8 with 0s
    if len(beta) < 8:
        beta = np.pad(beta, (0, 8 - len(beta)), mode="constant")
    beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7 = beta.tolist()

    # Function to integrate
    def integrand_1(x1, x2, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7):
        return (
            logistic(
                beta0
                + beta1
                + beta2 * x1
                + beta3 * x2
                + beta4 * x1 * x2
                + beta5 * x1**2
                + beta6 * x2**2
                + beta7 * 1
            )
            * norm.pdf(x1)
            * norm.pdf(x2)
        )

    def integrand_0(x1, x2, beta0, beta2, beta3, beta4, beta5, beta6, beta7):
        return (
            logistic(
                beta0
                + beta2 * x1
                + beta3 * x2
                + beta4 * x1 * x2
                + beta5 * x1**2
                + beta6 * x2**2
                + beta7 * 0
            )
            * norm.pdf(x1)
            * norm.pdf(x2)
        )

    # Double integration over normal distributions for X1 and X2 within finite bounds
    integration_bounds = (-10, 10)
    result_1, _ = dblquad(
        lambda x1, x2: integrand_1(
            x1, x2, beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7
        ),
        integration_bounds[0],
        integration_bounds[1],
        lambda x: integration_bounds[0],
        lambda x: integration_bounds[1],
    )
    result_0, _ = dblquad(
        lambda x1, x2: integrand_0(
            x1, x2, beta_0, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7
        ),
        integration_bounds[0],
        integration_bounds[1],
        lambda x: integration_bounds[0],
        lambda x: integration_bounds[1],
    )

    # Theoretical ATE
    ATE_correct = result_1 - result_0

    return ATE_correct
