import unittest
from typing import List

import numpy as np
from scipy.special import expit

from CausalEstimate.simulation.binary_simulation import (
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
    simulate_binary_data,
)


class TestEffectBase(unittest.TestCase):
    n: int = 10000
    alpha: List[float] = [0.1, 0.2, -0.3, 0]
    beta: List[float] = [0.5, 0.8, -0.6, 0.3, 0]
    seed: int = 42

    @classmethod
    def setUpClass(cls):
        # Simulate realistic data for testing
        rng = np.random.default_rng(cls.seed)
        # Covariates
        data = simulate_binary_data(
            cls.n, alpha=cls.alpha, beta=cls.beta, seed=cls.seed
        )

        # Predicted outcomes
        X = data[["X1", "X2"]].values
        A = data["A"].values
        Y = data["Y"].values
        ps = expit(
            cls.alpha[0] + cls.alpha[1] * X[:, 0] + cls.alpha[2] * X[:, 1]
        ) + 0.01 * rng.normal(size=cls.n)
        Y1_hat = expit(
            cls.beta[0]
            + cls.beta[1] * 1
            + cls.beta[2] * X[:, 0]
            + cls.beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=cls.n)
        Y0_hat = expit(
            cls.beta[0] + cls.beta[2] * X[:, 0] + cls.beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=cls.n)
        Yhat = expit(
            cls.beta[0]
            + cls.beta[1] * A
            + cls.beta[2] * X[:, 0]
            + cls.beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=cls.n)

        # clip
        eps = 1e-6
        Yhat = np.clip(Yhat, eps, 1 - eps)
        Y1_hat = np.clip(Y1_hat, eps, 1 - eps)
        Y0_hat = np.clip(Y0_hat, eps, 1 - eps)
        ps = np.clip(ps, eps, 1 - eps)

        cls.A = A
        cls.Y = Y
        cls.ps = ps
        cls.Y1_hat = Y1_hat
        cls.Y0_hat = Y0_hat
        cls.Yhat = Yhat

        true_ate = compute_ATE_theoretical_from_data(data, beta=cls.beta)
        true_att = compute_ATT_theoretical_from_data(data, beta=cls.beta)
        cls.true_ate = true_ate
        cls.true_att = true_att

        # for classes that take dataframe as input
        cls.data = data
        cls.data["PID"] = np.arange(len(data))
        cls.data["treatment"] = A
        cls.data["outcome"] = Y
        cls.data["ps"] = ps
        cls.data["Y1_hat"] = Y1_hat
        cls.data["Y0_hat"] = Y0_hat
        cls.data["Yhat"] = Yhat
