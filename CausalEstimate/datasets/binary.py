from typing import Optional, Tuple, Union

import pandas as pd

from CausalEstimate.simulation.binary_simulation import (
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
    compute_RR_theoretical_from_data,
    simulate_binary_data,
)
from CausalEstimate.utils.constants import OUTCOME_CF_COL

# default parameters for a reasonable simulation scenario
ALPHA = [0.1, 0.3, -0.2, 0.5]  # treatment model parameters
BETA = [0.2, 0.8, 0.4, -0.3, 0.6]  # outcome model parameters


def _true_effects(data: pd.DataFrame, beta: list) -> dict:
    return {
        "true_ate": compute_ATE_theoretical_from_data(data, beta),
        "true_att": compute_ATT_theoretical_from_data(data, beta),
        "true_rr": compute_RR_theoretical_from_data(data, beta),
    }


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
        The generated dataset with columns ['X1', 'X2', 'treatment', 'Y', 'Y_cf']
    params : dict, optional
        Dictionary containing the true parameters and true effects
        (true_ate, true_att, true_rr) if return_params=True

    Examples
    --------
    >>> from CausalEstimate.datasets import load_binary
    >>> data = load_binary(n_samples=1000, random_state=42)
    >>> data.shape
    (1000, 5)
    """
    data = simulate_binary_data(n=n_samples, alpha=ALPHA, beta=BETA, seed=random_state)

    if return_params:
        params = {
            "treatment_params": ALPHA,
            "outcome_params": BETA,
            **_true_effects(data, BETA),
            "DESCR": """
            Synthetic binary treatment-outcome dataset.

            The data is generated using a logistic model for both treatment assignment
            and outcome. Features X1 and X2 are drawn from standard normal distributions.

            Treatment model (logit):
            logit(P(A=1)) = α₀ + α₁X₁ + α₂X₂ + α₃X₁X₂

            Outcome model (logit):
            logit(P(Y=1)) = β₀ + β₁A + β₂X₁ + β₃X₂ + β₄X₁X₂

            True effects computed from the drawn sample:
            true_ate, true_att, true_rr
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
        - treatment: treatment assignment
        - Y: observed outcome
        - ps: true propensity scores
        - probas: outcome probabilities under the assigned treatment
        - probas_t0: outcome probabilities under control
        - probas_t1: outcome probabilities under treatment
    params : dict, optional
        Dictionary containing the true parameters and true effects
        (true_ate, true_att, true_rr) if return_params=True

    Examples
    --------
    >>> from CausalEstimate.datasets import load_binary_with_probas
    >>> data = load_binary_with_probas(n_samples=1000, random_state=42)
    >>> data.shape
    (1000, 8)
    """
    data = simulate_binary_data(
        n=n_samples, alpha=ALPHA, beta=BETA, seed=random_state, return_probas=True
    ).drop(columns=[OUTCOME_CF_COL])

    if return_params:
        params = {
            "treatment_params": ALPHA,
            "outcome_params": BETA,
            **_true_effects(data, BETA),
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
            - probas: outcome probabilities under the assigned treatment
            - probas_t0: outcome probabilities under control
            - probas_t1: outcome probabilities under treatment

            True effects computed from the drawn sample:
            true_ate, true_att, true_rr
            """,
        }
        return data, params

    return data
