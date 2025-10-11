import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.special import expit

from CausalEstimate.simulation.binary_simulation import (
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
    compute_RR_theoretical_from_data,
    simulate_binary_data,
)
from CausalEstimate.utils.constants import (
    OUTCOME_COL,
    PID_COL,
    PROBAS_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
    PS_COL,
    TREATMENT_COL,
)


def generate_simulation_data(
    n: int, alpha: List[float], beta: List[float], noise_level: float, seed: int
) -> Dict[str, Any]:
    """
    Generates a single, fresh dataset for a simulation run.
    This is the logic extracted from TestEffectBase.setUpClass.
    """
    rng = np.random.default_rng(seed)

    # 1. Simulate data using the true DGP
    data = simulate_binary_data(n, alpha=alpha, beta=beta, seed=seed)
    X_raw = data[["X1", "X2"]].values
    A = data[TREATMENT_COL].values
    Y = data[OUTCOME_COL].values

    # 2. Generate nuisance predictions using the assumed (misspecified) model
    ps_model_coeffs = np.array(alpha[:3])
    X_ps_design = np.column_stack([np.ones(n), X_raw])
    ps_logit = X_ps_design @ ps_model_coeffs + noise_level * rng.normal(size=n)
    ps = expit(ps_logit)

    outcome_model_coeffs = np.array(beta[:4])
    X_y1_design = np.column_stack([np.ones(n), np.ones(n), X_raw])
    X_y0_design = np.column_stack([np.ones(n), np.zeros(n), X_raw])
    X_y_obs_design = np.column_stack([np.ones(n), A, X_raw])

    Y1_hat = expit(
        X_y1_design @ outcome_model_coeffs + noise_level * rng.normal(size=n)
    )
    Y0_hat = expit(
        X_y0_design @ outcome_model_coeffs + noise_level * rng.normal(size=n)
    )
    Yhat = expit(
        X_y_obs_design @ outcome_model_coeffs + noise_level * rng.normal(size=n)
    )

    # 3. Finalize data and compute true values
    eps = 1e-7

    return {
        "A": A,
        "Y": Y,
        "ps": np.clip(ps, eps, 1 - eps),
        "Y1_hat": np.clip(Y1_hat, eps, 1 - eps),
        "Y0_hat": np.clip(Y0_hat, eps, 1 - eps),
        "Yhat": np.clip(Yhat, eps, 1 - eps),
        "true_ate": compute_ATE_theoretical_from_data(data, beta=beta),
        "true_att": compute_ATT_theoretical_from_data(data, beta=beta),
        "true_rr": compute_RR_theoretical_from_data(data, beta=beta),
    }


class TestEffectBase(unittest.TestCase):
    """
    Base class for single-run tests. It now uses the centralized
    generate_simulation_data function to create its fixture.
    """

    n: int = 30_000
    alpha: list = [0.1, 0.2, -0.3]
    beta: list = [0.5, 0.8, -0.6, 0.3]
    noise_level: float = 0
    seed: int = 41

    @classmethod
    def setUpClass(cls):
        """
        Generate a single, large dataset to be used as a fixture
        for all the simple, single-run estimator tests.
        """
        sim_data = generate_simulation_data(
            n=cls.n,
            alpha=cls.alpha,
            beta=cls.beta,
            noise_level=cls.noise_level,
            seed=cls.seed,
        )

        # Unpack the dictionary into class attributes for the tests to use
        cls.A = sim_data["A"]
        cls.Y = sim_data["Y"]
        cls.ps = sim_data["ps"]
        cls.Y1_hat = sim_data["Y1_hat"]
        cls.Y0_hat = sim_data["Y0_hat"]
        cls.Yhat = sim_data["Yhat"]
        cls.true_ate = sim_data["true_ate"]
        cls.true_att = sim_data["true_att"]
        cls.true_rr = sim_data["true_rr"]

        cls.data = pd.DataFrame(
            {
                TREATMENT_COL: cls.A,
                OUTCOME_COL: cls.Y,
                PS_COL: cls.ps,
                PROBAS_T1_COL: cls.Y1_hat,
                PROBAS_T0_COL: cls.Y0_hat,
                PROBAS_COL: cls.Yhat,
                PID_COL: np.arange(len(cls.A)),
            }
        )
