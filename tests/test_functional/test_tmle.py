import unittest

import numpy as np
from scipy.special import expit

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    estimate_fluctuation_parameter,
    update_ate_estimate,
)
from CausalEstimate.simulation.binary_simulation import (
    simulate_binary_data,
    compute_ATE_theoretical_from_data,
)


class TestTMLEFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Simulate realistic data for testing
        rng = np.random.default_rng(42)
        n = 2000
        # Covariates
        data = simulate_binary_data(
            n, alpha=[0.1, 0.2, -0.3, 0], beta=[0.5, 0.8, -0.6, 0.3, 0], seed=42
        )
        true_ate = compute_ATE_theoretical_from_data(
            data, beta=[0.5, 0.8, -0.6, 0.3, 0]
        )

        # Predicted outcomes
        X = data[["X1", "X2"]].values
        A = data["A"].values
        Y = data["Y"].values
        ps = expit(0.1 + 0.2 * X[:, 0] - 0.3 * X[:, 1]) + 0.01 * rng.normal(size=n)
        Y1_hat = expit(
            0.5 + 0.8 * 1 + -0.6 * X[:, 0] + 0.3 * X[:, 1]
        ) + 0.01 * rng.normal(size=n)
        Y0_hat = expit(0.5 + -0.6 * X[:, 0] + 0.3 * X[:, 1]) + 0.01 * rng.normal(size=n)
        Yhat = expit(
            0.5 + 0.8 * A + -0.6 * X[:, 0] + 0.3 * X[:, 1]
        ) + 0.01 * rng.normal(size=n)

        cls.A = A
        cls.Y = Y
        cls.ps = ps
        cls.Y1_hat = Y1_hat
        cls.Y0_hat = Y0_hat
        cls.Yhat = Yhat
        cls.true_ate = true_ate

    def test_estimate_fluctuation_parameter(self):
        epsilon = estimate_fluctuation_parameter(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))

    def test_update_ate_estimate(self):
        epsilon = 0.1  # Arbitrary small fluctuation parameter
        ate = update_ate_estimate(self.ps, self.Y0_hat, self.Y1_hat, epsilon)
        self.assertIsInstance(ate, float)
        # Check that ate is within a reasonable range
        self.assertTrue(-5 <= ate <= 5)

    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertIsInstance(ate_tmle, float)
        # The true ATE is 2; check if the estimate is close
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.1)

    def test_compute_tmle_ate_edge_cases(self):
        # Test with ps very close to 0 or 1
        ps_edge = self.ps.copy()
        ps_edge[ps_edge < 0.01] = 0.01
        ps_edge[ps_edge > 0.99] = 0.99
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, ps_edge, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertIsInstance(ate_tmle, float)
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.15)


if __name__ == "__main__":
    unittest.main()
