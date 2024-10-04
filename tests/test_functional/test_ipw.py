import unittest
import numpy as np
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_ate_stabilized,
    compute_ipw_att,
    compute_ipw_risk_ratio,
    compute_ipw_risk_ratio_treated,
)


class TestIPWEstimators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulate simple data for testing
        rng = np.random.default_rng(42)
        n = 1000
        cls.A = rng.binomial(1, 0.5, size=n)  # Treatment assignment
        cls.Y = rng.binomial(1, 0.3, size=n)  # Outcome
        cls.ps = np.clip(rng.uniform(0.1, 0.9, size=n), 0.01, 0.99)  # Propensity score

    def test_ipw_ate(self):
        ate = compute_ipw_ate(self.A, self.Y, self.ps)
        self.assertIsInstance(ate, float)
        self.assertTrue(-1 <= ate <= 1)  # Check ATE is within reasonable range

    def test_ipw_ate_stabilized(self):
        ate_stabilized = compute_ipw_ate_stabilized(self.A, self.Y, self.ps)
        self.assertIsInstance(ate_stabilized, float)
        self.assertTrue(-1 <= ate_stabilized <= 1)  # Check ATE with stabilized weights

    def test_ipw_att(self):
        att = compute_ipw_att(self.A, self.Y, self.ps)
        self.assertIsInstance(att, float)
        self.assertTrue(-1 <= att <= 1)  # Check ATT is within reasonable range

    def test_ipw_risk_ratio(self):
        risk_ratio = compute_ipw_risk_ratio(self.A, self.Y, self.ps)
        self.assertIsInstance(risk_ratio, float)
        self.assertTrue(risk_ratio > 0)  # Risk ratio should be positive

    def test_ipw_risk_ratio_treated(self):
        risk_ratio_treated = compute_ipw_risk_ratio_treated(self.A, self.Y, self.ps)
        self.assertIsInstance(risk_ratio_treated, float)
        self.assertTrue(
            risk_ratio_treated > 0
        )  # Risk ratio for treated should be positive

    def test_edge_case_ps_near_0_or_1(self):
        # Test with ps values close to 0 or 1
        ps_edge = np.clip(self.ps, 0.01, 0.99)
        ate_edge = compute_ipw_ate(self.A, self.Y, ps_edge)
        self.assertIsInstance(ate_edge, float)
        self.assertTrue(-1 <= ate_edge <= 1)

        att_edge = compute_ipw_att(self.A, self.Y, ps_edge)
        self.assertIsInstance(att_edge, float)
        self.assertTrue(-1 <= att_edge <= 1)

    def test_mismatched_shapes(self):
        # Test with mismatched input shapes
        A = np.array([1, 0, 1])
        Y = np.array([3, 1, 4])
        ps = np.array([0.8, 0.6])  # Mismatched length

        with self.assertRaises(ValueError):
            compute_ipw_ate(A, Y, ps)

    def test_single_value_input(self):
        # Test with single value input
        A = np.array([1])
        Y = np.array([1])
        ps = np.array([0.5])

        ate = compute_ipw_ate(A, Y, ps)
        self.assertIsInstance(ate, float)


# Run the unittests
if __name__ == "__main__":
    unittest.main()
