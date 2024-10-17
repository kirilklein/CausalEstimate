import unittest
import numpy as np
from CausalEstimate.estimators.functional.aipw import compute_aipw_ate, compute_aipw_att

from tests.test_functional.base import TestEffectBase


class TestComputeAIPWATE(TestEffectBase):
    def test_aipw_ate_computation(self):
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

class TestAIPW_ATE_ps_misspecified(TestEffectBase):
    alpha = [0.1, 0.2, -0.3, 1]
    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(ate_aipw, float)
        # Check if the estimate is still close to the true ATE
        self.assertAlmostEqual(ate_aipw, self.true_ate, delta=0.1)

class TestAIPW_ATE_outcome_model_misspecified(TestEffectBase):
    beta = [0.5, 0.8, -0.6, 0.3, 5]
    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(ate_aipw, float)
        # Check if the estimate is still close to the true ATE
        self.assertAlmostEqual(ate_aipw, self.true_ate, delta=0.1)

class TestComputeAIPWATT(TestEffectBase):
    def test_aipw_att_computation(self):
        # Test the computation of AIPW ATT
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(att_aipw, float)
        # Check if the AIPW estimate is close to the true ATT
        self.assertAlmostEqual(att_aipw, self.true_att, delta=0.1)

class TestComputeAIPWATT_outcome_model_misspecified(TestEffectBase):
    beta = [0.5, 0.8, -0.6, 0.3, 5]
    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(att_aipw, float)
        # Check if the estimate is still close to the true ATT
        self.assertAlmostEqual(att_aipw, self.true_att, delta=0.1)

class TestComputeAIPWATT_ps_misspecified(TestEffectBase):
    alpha = [0.1, 0.2, -0.3, 1]
    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertIsInstance(att_aipw, float)
        # Check if the estimate is still close to the true ATT
        self.assertAlmostEqual(att_aipw, self.true_att, delta=0.1)


# Run the unittests
if __name__ == "__main__":
    unittest.main()
