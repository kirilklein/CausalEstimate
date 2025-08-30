import unittest
from typing import List

import numpy as np
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


class TestEffectBase(unittest.TestCase):
    """
    Base class for testing causal effect estimators.

    The TRUE data generating process (DGP) is controlled by the full `alpha` and
    `beta` vectors. Child classes can override these to create different DGPs.

    - alpha: [intercept, X1, X2, X1*X2 interaction]
    - beta:  [intercept, A, X1, X2, X1*X2 interaction]

    The predictions for ps, Y1_hat, and Y0_hat are generated from an *assumed model*
    that only uses the main effects (the first 3 or 4 coefficients). This is done
    intentionally, so that if a child class sets a non-zero interaction term in
    `alpha` or `beta`, the model used for predictions becomes misspecified.
    The default values are set to a correctly specified DGP.
    """

    n: int = 30_000
    alpha: List[float] = [0.1, 0.2, -0.3]
    beta: List[float] = [0.5, 0.8, -0.6, 0.3]
    noise_level: float = 0  # logit
    cutoff_epsilon: float = 1e-7
    seed: int = 41

    @classmethod
    def setUpClass(cls):
        # Simulate realistic data for testing
        rng = np.random.default_rng(cls.seed)

        # 1. Simulate data using the full coefficient vectors (the TRUE DGP).
        #    Child classes override cls.alpha/cls.beta to change this DGP.
        data = simulate_binary_data(
            cls.n, alpha=cls.alpha, beta=cls.beta, seed=cls.seed
        )

        X_raw = data[["X1", "X2"]].values
        A = data[TREATMENT_COL].values
        Y = data[OUTCOME_COL].values

        # 2. Generate predictions using an ASSUMED model that only considers
        #    main effects. This is where the misspecification is introduced.

        # --- Propensity Score Model ---
        # The assumed model for PS uses only the first 3 coefficients (intercept, X1, X2).
        # If `cls.alpha` has a non-zero 4th element, this model is MISSPECIFIED.
        ps_model_coeffs = np.array(cls.alpha[:3])
        X_ps_design = np.column_stack(
            [np.ones(cls.n), X_raw]
        )  # Design matrix for main effects
        ps = expit(X_ps_design @ ps_model_coeffs) + cls.noise_level * rng.normal(
            size=cls.n
        )

        outcome_model_coeffs = np.array(cls.beta[:4])

        # Design matrices for main effects outcome model
        X_y1_design = np.column_stack([np.ones(cls.n), np.ones(cls.n), X_raw])  # A=1
        X_y0_design = np.column_stack([np.ones(cls.n), np.zeros(cls.n), X_raw])  # A=0
        X_y_obs_design = np.column_stack([np.ones(cls.n), A, X_raw])  # A=observed

        # Generate predictions for Y1_hat, Y0_hat, and Yhat
        Y1_hat = expit(
            X_y1_design @ outcome_model_coeffs
        ) + cls.noise_level * rng.normal(size=cls.n)

        Y0_hat = expit(
            X_y0_design @ outcome_model_coeffs
        ) + cls.noise_level * rng.normal(size=cls.n)
        Yhat = expit(
            X_y_obs_design @ outcome_model_coeffs
        ) + cls.noise_level * rng.normal(size=cls.n)

        # 3. Finalize data preparation
        eps = cls.cutoff_epsilon
        cls.A, cls.Y = A, Y
        cls.ps = np.clip(ps, eps, 1 - eps)
        cls.Y1_hat = np.clip(Y1_hat, eps, 1 - eps)
        cls.Y0_hat = np.clip(Y0_hat, eps, 1 - eps)
        cls.Yhat = np.clip(Yhat, eps, 1 - eps)

        cls.true_ate = compute_ATE_theoretical_from_data(data, beta=cls.beta)
        cls.true_att = compute_ATT_theoretical_from_data(data, beta=cls.beta)
        cls.true_rr = compute_RR_theoretical_from_data(data, beta=cls.beta)

        # for classes that take dataframe as input
        cls.data = data
        cls.data[PID_COL] = np.arange(len(data))
        cls.data[TREATMENT_COL] = A
        cls.data[OUTCOME_COL] = Y
        cls.data[PS_COL] = ps
        cls.data[PROBAS_T1_COL] = Y1_hat
        cls.data[PROBAS_T0_COL] = Y0_hat
        cls.data[PROBAS_COL] = Yhat
