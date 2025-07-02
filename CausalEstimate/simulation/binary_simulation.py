import numpy as np
import pandas as pd
from scipy.special import expit as logistic

from CausalEstimate.utils.constants import OUTCOME_CF_COL, OUTCOME_COL, TREATMENT_COL


def simulate_binary_data(n: int, alpha: list, beta: list, seed=None) -> pd.DataFrame:
    """
    Generate a synthetic dataset with two continuous covariates, a binary treatment,
    and potential binary outcomes under both observed and counterfactual treatments.

    Parameters
    ----------
    n : int
        Number of samples to simulate.
    alpha : sequence of float
        Coefficients for the treatment assignment model (logistic):
        - alpha[0]: intercept
        - alpha[1]: linear effect of X1
        - alpha[2]: linear effect of X2
        - alpha[3]: interaction effect X1 * X2
        - alpha[4]: quadratic effect X1²  (optional; defaults to 0 if not provided)
        - alpha[5]: quadratic effect X2²  (optional; defaults to 0 if not provided)
        If `len(alpha) < 6`, remaining coefficients are padded with zeros.
    beta : sequence of float
        Coefficients for the potential outcome model (logistic):
        - beta[0]: intercept
        - beta[1]: main effect of treatment A
        - beta[2]: linear effect of X1
        - beta[3]: linear effect of X2
        - beta[4]: interaction effect X1 * X2
        - beta[5]: quadratic effect X1²  (optional; defaults to 0)
        - beta[6]: quadratic effect X2²  (optional; defaults to 0)
        - beta[7]: optional squared‐treatment term A² (defaults to 0)
        If `len(beta) < 8`, remaining coefficients are padded with zeros.
    seed : int or None, optional
        Random seed for reproducibility. If None (default), a random generator
        without a fixed seed is used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - "X1", "X2": continuous covariates ~ N(0,1)
        - `<TREATMENT_COL>`: binary treatment assignment A ∼ Bernoulli(sigmoid(alpha⋅[1, X1, X2, …]))
        - `<OUTCOME_COL>`: observed binary outcome Y ∼ Bernoulli(sigmoid(beta⋅[A, X1, X2, …]))
        - `<OUTCOME_CF_COL>`: counterfactual outcome Y_cf under the *opposite* treatment A_cf = 1−A.

    Notes
    -----
    - Uses a logistic link (`scipy.special.expit`) internally.
    - Quadratic and interaction terms beyond the first four coefficients are
      optional and default to zero if omitted.
    - Counterfactual outcomes are generated with the same coefficient set,
      simply flipping the treatment value.
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
        beta[0]
        + beta[1] * (1 - A)
        + beta[2] * X1
        + beta[3] * X2
        + beta[4] * X1 * X2
        + beta[5] * X1**2
        + beta[6] * X2**2
        + beta[7] * (1 - A) ** 2
    )
    q_cf = logistic(logit_q_cf)
    Y_cf = rng.binomial(1, q_cf)

    data = pd.DataFrame(
        {"X1": X1, "X2": X2, TREATMENT_COL: A, OUTCOME_COL: Y, OUTCOME_CF_COL: Y_cf}
    )

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
    treated_data = data[data[TREATMENT_COL] == 1]
    E_Y1 = compute_expected_outcome(treated_data, beta, 1)
    E_Y0 = compute_expected_outcome(treated_data, beta, 0)
    return E_Y1 - E_Y0


def compute_RR_theoretical_from_data(data: pd.DataFrame, beta: list):
    """Compute the true risk ratio (RR) from the model coefficients, using the data."""
    E_Y1 = compute_expected_outcome(data, beta, 1)
    E_Y0 = compute_expected_outcome(data, beta, 0)
    return E_Y1 / E_Y0
