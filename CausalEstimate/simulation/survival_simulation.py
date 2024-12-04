"""
Simulate survival data for testing time-to-event estimators.

This module implements the data generating process from:
Cai, Weixin, and Mark J. van der Laan. "One-step targeted maximum likelihood estimation 
for time-to-event outcomes." Biometrics 76.3 (2020): 722-733.

The simulation creates:
- Covariates (X): Uniform distribution
- Treatment (A): Binary treatment dependent on X
- Time-to-event (T): Exponential distribution with rate depending on X and A
- Censoring time (C): Weibull distribution depending on X
- Observed outcome: Minimum of T and C

Note: The implementation differs slightly from the paper's description, which likely 
contains a mistake in the data generating process. We use simulation they have in the R code.
"""

import numpy as np
import pandas as pd

# Simulation parameters grouped by component
COVARIATE_PARAMS = {"MIN": 0, "MAX": 1.5}  # Minimum value for X  # Maximum value for X

TREATMENT_PARAMS = {
    "OFFSET": 0.4,  # Base probability of treatment
    "MULT": 0.5,  # Multiplier for X effect on treatment
    "MIN_X": 0.75,  # Threshold for X effect on treatment
}

OUTCOME_PARAMS = {
    "OFFSET": 1,  # Base rate for exponential distribution
    "X_MULT": 0.7,  # Multiplier for X effect on outcome
    "A_MULT": 0.8,  # Multiplier for treatment effect
}

CENSORING_PARAMS = {
    "OFFSET": 1,  # Base shape for Weibull distribution
    "MULT": 0.5,  # Multiplier for X effect on censoring
    "SCALE": 75,  # Scale parameter for Weibull distribution
}

def simulate_data(
    n: int,
    A: np.ndarray = None,
    covariate_kwargs: dict = {},
    treatment_kwargs: dict = {},
    censoring_kwargs: dict = {},
    outcome_kwargs: dict = {},
) -> pd.DataFrame:
    """
    Simulate complete survival dataset.

    Args:
        n: Number of samples
        A: Optional fixed treatment assignments
        *_kwargs: Optional parameters for each simulation component

    Returns:
        DataFrame containing:
            - X: Covariates
            - A: Treatment assignments
            - T: True event times
            - C: Censoring times
            - T_observed: Observed times (min of T and C)
            - Y: Event indicators (1 if T <= C)
            - pid: Sample IDs
    """
    np.random.seed(42)

    # Generate data components
    X = simulate_covariates(n, **covariate_kwargs)
    if A is None:
        A = simulate_treatment(n, X, **treatment_kwargs)
    else:
        A = np.ones(n) * A

    T = simulate_outcome_time(n, X, A, **outcome_kwargs)
    C = simulate_censoring(n, X, **censoring_kwargs)

    # Compute observed outcomes
    T_observed = np.minimum(T, C)
    Y = (T <= C).astype(int)

    # Create and return dataset
    data = pd.DataFrame(
        {
            "X": X,
            "A": A,
            "T": T,
            "C": C,
            "T_observed": T_observed,
            "Y": Y,
            "pid": np.arange(n),
        }
    )
    return data


def simulate_covariates(
    n: int, min: float = COVARIATE_PARAMS["MIN"], max: float = COVARIATE_PARAMS["MAX"]
) -> np.ndarray:
    """
    Generate covariates from uniform distribution.

    Args:
        n: Number of samples
        min: Minimum value for uniform distribution
        max: Maximum value for uniform distribution

    Returns:
        Array of covariates X
    """
    return np.random.uniform(min, max, n)


def simulate_treatment(
    n: int,
    X: np.ndarray,
    offset: float = TREATMENT_PARAMS["OFFSET"],
    multiplier: float = TREATMENT_PARAMS["MULT"],
    min_X: float = TREATMENT_PARAMS["MIN_X"],
) -> np.ndarray:
    """
    Generate binary treatment assignments based on covariates.

    Args:
        n: Number of samples
        X: Covariate values
        offset: Base probability of treatment
        multiplier: Effect of X on treatment probability
        min_X: Threshold for X effect

    Returns:
        Binary array of treatment assignments
    """
    prob = offset + multiplier * (X > min_X)
    return np.random.binomial(1, prob, n)


def compute_exponential_rate(
    X: np.ndarray,
    A: np.ndarray,
    rate_offset: float = OUTCOME_PARAMS["OFFSET"],
    rate_X_mult: float = OUTCOME_PARAMS["X_MULT"],
    rate_A_mult: float = OUTCOME_PARAMS["A_MULT"],
) -> np.ndarray:
    """
    Compute rate parameter for exponential outcome distribution.

    Rate = offset + X_mult * X² - A_mult * A
    """
    return rate_offset + rate_X_mult * X**2 - rate_A_mult * A


def simulate_outcome_time(
    n: int,
    X: np.ndarray,
    A: np.ndarray,
    rate_offset: float = OUTCOME_PARAMS["OFFSET"],
    rate_X_mult: float = OUTCOME_PARAMS["X_MULT"],
    rate_A_mult: float = OUTCOME_PARAMS["A_MULT"],
) -> np.ndarray:
    """
    Generate time-to-event outcomes from exponential distribution.

    The rate parameter depends on both covariates and treatment.
    Times are rounded and scaled by 2.
    """
    rate = compute_exponential_rate(X, A, rate_offset, rate_X_mult, rate_A_mult)
    T_exp = np.random.exponential(1 / rate, n)
    return np.round(T_exp * 2)


def simulate_censoring(
    n: int,
    X: np.ndarray,
    shape_offset: float = CENSORING_PARAMS["OFFSET"],
    shape_multiplier: float = CENSORING_PARAMS["MULT"],
    scale: float = CENSORING_PARAMS["SCALE"],
) -> np.ndarray:
    """
    Generate censoring times from Weibull distribution.

    The shape parameter depends on covariates.
    Times are rounded and scaled by 2.
    """
    shape = shape_offset + shape_multiplier * X
    C_weib = np.random.weibull(shape, n) * scale
    return np.round(C_weib * 2)


def theoretical_survival_function(
    t: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    rate_offset: float = OUTCOME_PARAMS["OFFSET"],
    rate_X_mult: float = OUTCOME_PARAMS["X_MULT"],
    rate_A_mult: float = OUTCOME_PARAMS["A_MULT"],
) -> np.ndarray:
    """
    Compute theoretical survival probabilities for given times.

    Args:
        t: Array of time points
        X: Covariates
        A: Treatment assignments

    Returns:
        Matrix of survival probabilities (n_samples × n_timepoints)
    """
    rate = compute_exponential_rate(X, A, rate_offset, rate_X_mult, rate_A_mult)
    rate = np.asarray(rate)
    survival_prob = np.exp(-rate[:, np.newaxis] * t / 2)
    return survival_prob
