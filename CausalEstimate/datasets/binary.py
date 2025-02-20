from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from CausalEstimate.utils.constants import (
    TREATMENT_COL,
    OUTCOME_COL,
    PS_COL,
    PROBAS_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
)


def load_binary(
    n_samples: int = 1000,
    random_state: Optional[int] = None,
    return_params: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Load and return a synthetic binary treatment-outcome dataset.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    random_state : int or None, default=None
        Random state for reproducibility
    return_params : bool, default=False
        If True, returns the true parameters used to generate the data

    Returns
    -------
    X : pandas.DataFrame
        The generated dataset with columns ['X1', 'X2', 'A', 'Y', 'Y_cf']
    params : dict, optional
        Dictionary containing the true parameters if return_params=True

    Examples
    --------
    >>> from CausalEstimate.datasets import load_binary
    >>> data = load_binary(n_samples=1000, random_state=42)
    >>> data.shape
    (1000, 5)
    """
    # Define default parameters for a reasonable simulation scenario
    alpha = [0.1, 0.3, -0.2, 0.5]  # treatment model parameters
    beta = [0.2, 0.8, 0.4, -0.3, 0.6]  # outcome model parameters

    data = simulate_binary_data(n=n_samples, alpha=alpha, beta=beta, seed=random_state)

    if return_params:
        params = {
            "treatment_params": alpha,
            "outcome_params": beta,
            "DESCR": """
            Synthetic binary treatment-outcome dataset.
            
            The data is generated using a logistic model for both treatment assignment
            and outcome. Features X1 and X2 are drawn from standard normal distributions.
            
            Treatment model (logit):
            logit(P(A=1)) = α₀ + α₁X₁ + α₂X₂ + α₃X₁X₂
            
            Outcome model (logit):
            logit(P(Y=1)) = β₀ + β₁A + β₂X₁ + β₃X₂ + β₄X₁X₂
            """,
        }
        return data, params

    return data


def load_binary_with_probas(
    n_samples: int = 1000,
    random_state: Optional[int] = None,
    return_params: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Load and return a synthetic binary treatment-outcome dataset with probabilities.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    random_state : int or None, default=None
        Random state for reproducibility
    return_params : bool, default=False
        If True, returns the true parameters used to generate the data

    Returns
    -------
    X : pandas.DataFrame
        The generated dataset with columns:
        - X1, X2: covariates
        - A: treatment assignment
        - Y: observed outcome
        - ps: propensity scores
        - Y_prob: outcome probabilities
        - Y_cf_0: counterfactual outcome probabilities under control
        - Y_cf_1: counterfactual outcome probabilities under treatment
    params : dict, optional
        Dictionary containing the true parameters if return_params=True

    Examples
    --------
    >>> from CausalEstimate.datasets import load_binary_with_probas
    >>> data = load_binary_with_probas(n_samples=1000, random_state=42)
    >>> data.shape
    (1000, 8)
    """
    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Define default parameters for a reasonable simulation scenario
    alpha = [0.1, 0.3, -0.2, 0.5]  # treatment model parameters
    beta = [0.2, 0.8, 0.4, -0.3, 0.6]  # outcome model parameters

    # Generate covariates
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X1X2 = X1 * X2

    # Generate propensity scores
    ps_logit = alpha[0] + alpha[1] * X1 + alpha[2] * X2 + alpha[3] * X1X2
    ps = 1 / (1 + np.exp(-ps_logit))
    ps = np.clip(ps, 0.01, 0.99)  # Ensure no extreme propensity scores

    # Generate treatment assignment
    A = np.random.binomial(1, ps)

    # Generate potential outcomes
    # Y(0): outcome probability under control
    Y_cf_0_logit = beta[0] + beta[2] * X1 + beta[3] * X2 + beta[4] * X1X2
    Y_cf_0 = 1 / (1 + np.exp(-Y_cf_0_logit))

    # Y(1): outcome probability under treatment
    Y_cf_1_logit = beta[0] + beta[1] + beta[2] * X1 + beta[3] * X2 + beta[4] * X1X2
    Y_cf_1 = 1 / (1 + np.exp(-Y_cf_1_logit))

    # Observed outcome probabilities based on treatment assignment
    Y_prob = np.where(A == 1, Y_cf_1, Y_cf_0)

    # Generate observed outcomes
    Y = np.random.binomial(1, Y_prob)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "X1": X1,
            "X2": X2,
            TREATMENT_COL: A,
            OUTCOME_COL: Y,
            PS_COL: ps,
            PROBAS_COL: Y_prob,
            PROBAS_T0_COL: Y_cf_0,
            PROBAS_T1_COL: Y_cf_1,
        }
    )

    if return_params:
        params = {
            "treatment_params": alpha,
            "outcome_params": beta,
            "DESCR": """
            Synthetic binary treatment-outcome dataset with probabilities.
            
            The data is generated using a logistic model for both treatment assignment
            and outcome. Features X1 and X2 are drawn from standard normal distributions.
            
            Treatment model (logit):
            logit(P(A=1)) = α₀ + α₁X₁ + α₂X₂ + α₃X₁X₂
            
            Outcome model (logit):
            logit(P(Y=1)) = β₀ + β₁A + β₂X₁ + β₃X₂ + β₄X₁X₂
            
            Additional columns:
            - ps: true propensity scores
            - Y_prob: observed outcome probabilities
            - Y_cf_0: counterfactual outcome probabilities under control
            - Y_cf_1: counterfactual outcome probabilities under treatment
            """,
        }
        return data, params

    return data
