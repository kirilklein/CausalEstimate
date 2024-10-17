import unittest
import numpy as np
from scipy.special import expit
from CausalEstimate.estimators.functional.aipw import compute_aipw_ate, compute_aipw_att
from CausalEstimate.simulation.binary_simulation import (
    simulate_binary_data,
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
)


class TestComputeAIPWATE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Simulate realistic data for testing
        rng = np.random.default_rng(42)
        n = 2000
        # Covariates
        alpha = [0.1, 0.2, -0.3, 0]
        beta = [0.5, 0.8, -0.6, 0.3, 0]
        data = simulate_binary_data(n, alpha=alpha, beta=beta, seed=42)
        true_ate = compute_ATE_theoretical_from_data(data, beta=beta)

        # Predicted outcomes
        X = data[["X1", "X2"]].values
        A = data["A"].values
        Y = data["Y"].values
        ps = expit(
            alpha[0] + alpha[1] * X[:, 0] + alpha[2] * X[:, 1]
        ) + 0.01 * rng.normal(size=n)
        Y1_hat = expit(
            beta[0] + beta[1] * 1 + beta[2] * X[:, 0] + beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=n)
        Y0_hat = expit(
            beta[0] + beta[2] * X[:, 0] + beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=n)

        cls.A = A
        cls.Y = Y
        cls.ps = ps
        cls.Y1_hat = Y1_hat
        cls.Y0_hat = Y0_hat
        cls.true_ate = true_ate

    def test_aipw_ate_computation(self):
        # Test the computation of AIPW ATE
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(ate_aipw, float)
        # Check if the AIPW estimate is close to the true ATE
        self.assertAlmostEqual(ate_aipw, self.true_ate, delta=0.1)

    def test_invalid_input_shapes(self):
        # Test for mismatched input shapes
        A = np.array([1, 0, 1])
        Y = np.array([3, 1, 4])
        ps = np.array([0.8, 0.6])  # Mismatched length
        Y0_hat = np.array([2, 1.5, 3])
        Y1_hat = np.array([3.5, 2.0, 4.5])

        # Ensure that an exception is raised for mismatched input shapes
        with self.assertRaises(ValueError):
            compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat)

    def test_edge_case_ps_close_to_zero_or_one(self):
        # Test with ps very close to 0 or 1
        ps_edge = self.ps.copy()
        ps_edge[ps_edge < 0.01] = 0.01
        ps_edge[ps_edge > 0.99] = 0.99

        # Compute the AIPW estimate with the edge case propensity scores
        ate_aipw = compute_aipw_ate(self.A, self.Y, ps_edge, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(ate_aipw, float)
        # Check if the estimate is still close to the true ATE
        self.assertAlmostEqual(ate_aipw, self.true_ate, delta=0.15)


class TestComputeAIPWATT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Simulate realistic data for testing
        rng = np.random.default_rng(42)
        n = 2000
        # Covariates
        alpha = [0.1, 0.2, -0.3, 0]
        beta = [0.5, 0.8, -0.6, 0.3, 0]
        data = simulate_binary_data(n, alpha=alpha, beta=beta, seed=42)
        true_att = compute_ATT_theoretical_from_data(data, beta=beta)

        # Predicted outcomes
        X = data[["X1", "X2"]].values
        A = data["A"].values
        Y = data["Y"].values
        ps = expit(
            alpha[0] + alpha[1] * X[:, 0] + alpha[2] * X[:, 1]
        ) + 0.01 * rng.normal(size=n)
        Y1_hat = expit(
            beta[0] + beta[1] * 1 + beta[2] * X[:, 0] + beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=n)
        Y0_hat = expit(
            beta[0] + beta[2] * X[:, 0] + beta[3] * X[:, 1]
        ) + 0.01 * rng.normal(size=n)

        cls.A = A
        cls.Y = Y
        cls.ps = ps
        cls.Y1_hat = Y1_hat
        cls.Y0_hat = Y0_hat
        cls.true_att = true_att

    def test_aipw_att_computation(self):
        # Test the computation of AIPW ATT
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(att_aipw, float)
        # Check if the AIPW estimate is close to the true ATT
        self.assertAlmostEqual(att_aipw, self.true_att, delta=0.1)


# Run the unittests
if __name__ == "__main__":
    unittest.main()
