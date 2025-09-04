import unittest

import numpy as np

from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_att,
    compute_ipw_risk_ratio,
    compute_ipw_risk_ratio_treated,
    compute_ipw_weights,
)
from CausalEstimate.utils.constants import EFFECT, EFFECT_untreated
from tests.helpers.setup import TestEffectBase


class TestIPWSanityChecks(unittest.TestCase):
    """
    Basic smoke tests and edge case handling for IPW estimators using random data.
    These tests ensure the functions run without crashing and produce plausible outputs.
    """

    @classmethod
    def setUpClass(cls):
        # Simulate simple data for testing
        rng = np.random.default_rng(42)
        n = 1000
        # Ensure both groups are present
        cls.A = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        cls.Y = rng.binomial(1, 0.3 + 0.2 * cls.A)  # Outcome now depends on treatment
        cls.ps = np.clip(rng.uniform(0.1, 0.9, size=n), 0.01, 0.99)

    def test_ipw_ate(self):
        ate = compute_ipw_ate(self.A, self.Y, self.ps)
        self.assertIsInstance(ate[EFFECT], float)
        self.assertTrue(-1 <= ate[EFFECT] <= 1)

    def test_ipw_att(self):
        att = compute_ipw_att(self.A, self.Y, self.ps)
        self.assertIsInstance(att[EFFECT], float)
        self.assertTrue(-1 <= att[EFFECT] <= 1)

    def test_empty_group_handling(self):
        # Test that providing data with only one group results in NaNs and warnings
        A_all_treated = np.ones(5)
        Y_all_treated = np.ones(5)
        ps_all_treated = np.full(5, 0.8)

        with self.assertWarns(RuntimeWarning):
            ate = compute_ipw_ate(A_all_treated, Y_all_treated, ps_all_treated)
            self.assertTrue(np.isnan(ate[EFFECT]))
            self.assertTrue(np.isnan(ate[EFFECT_untreated]))


class TestIPWEstimators(unittest.TestCase):
    """Basic tests for IPW estimators"""

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
        self.assertIsInstance(ate[EFFECT], float)
        self.assertTrue(-1 <= ate[EFFECT] <= 1)  # Check ATE is within reasonable range

    def test_ipw_att(self):
        att = compute_ipw_att(self.A, self.Y, self.ps)
        self.assertIsInstance(att[EFFECT], float)
        self.assertTrue(-1 <= att[EFFECT] <= 1)  # Check ATT is within reasonable range

    def test_ipw_risk_ratio(self):
        risk_ratio = compute_ipw_risk_ratio(self.A, self.Y, self.ps)
        self.assertIsInstance(risk_ratio[EFFECT], float)
        self.assertTrue(risk_ratio[EFFECT] > 0)  # Risk ratio should be positive

    def test_ipw_risk_ratio_treated(self):
        risk_ratio_treated = compute_ipw_risk_ratio_treated(self.A, self.Y, self.ps)
        self.assertIsInstance(risk_ratio_treated[EFFECT], float)
        self.assertTrue(
            risk_ratio_treated[EFFECT] > 0
        )  # Risk ratio for treated should be positive

    def test_edge_case_ps_near_0_or_1(self):
        # Test with ps values close to 0 or 1
        ps_edge = np.clip(self.ps, 0.01, 0.99)
        ate_edge = compute_ipw_ate(self.A, self.Y, ps_edge)
        self.assertIsInstance(ate_edge[EFFECT], float)
        self.assertTrue(-1 <= ate_edge[EFFECT] <= 1)

        att_edge = compute_ipw_att(self.A, self.Y, ps_edge)
        self.assertIsInstance(att_edge[EFFECT], float)
        self.assertTrue(-1 <= att_edge[EFFECT] <= 1)

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
        self.assertIsInstance(ate[EFFECT], float)


class TestIPWWeightFunction(unittest.TestCase):
    """
    Directly tests the `compute_ipw_weights` function to ensure logic is correct.
    """

    @classmethod
    def setUpClass(cls):
        cls.A = np.array([1, 1, 0, 0])
        cls.ps = np.array([0.8, 0.4, 0.5, 0.2])
        cls.pi = 0.5

    def test_att_weights(self):
        weights = compute_ipw_weights(self.A, self.ps, weight_type="ATT")
        stabilization_factor = (1 - self.pi) / self.pi
        expected = np.array(
            [
                1.0,
                1.0,
                (0.5 / 0.5) * stabilization_factor,
                (0.2 / 0.8) * stabilization_factor,
            ]
        )
        np.testing.assert_allclose(weights, expected)


# =============================================================================
# SECTION 2: Simulation-Based Tests
# These tests use the TestEffectBase to simulate data where the true effect is known.
# =============================================================================


class TestComputeIPW_base(TestEffectBase):
    def test_compute_ipw_ate(self):
        ate_ipw = compute_ipw_ate(self.A, self.Y, self.ps)
        self.assertAlmostEqual(ate_ipw[EFFECT], self.true_ate, delta=0.1)


class TestComputeIPWATE_ps_model_misspecified(TestComputeIPW_base):
    alpha = [0.1, 0.2, -0.3, 3]

    def test_compute_ipw_ate(self):
        ate_ipw = compute_ipw_ate(self.A, self.Y, self.ps)
        self.assertNotAlmostEqual(ate_ipw[EFFECT], self.true_ate, delta=0.02)


class TestComputeIPWATE_both_models_misspecified(TestComputeIPW_base):
    beta = [0.5, 0.8, -0.6, 0.3, 3]
    alpha = [0.1, 0.2, -0.3, 3]

    def test_compute_ipw_ate(self):
        ate_ipw = compute_ipw_ate(self.A, self.Y, self.ps)
        self.assertNotAlmostEqual(ate_ipw[EFFECT], self.true_ate, delta=0.1)


class TestComputeIPW_ATT(TestEffectBase):
    """Checks if IPW can recover the true ATT in a well-behaved simulation."""

    def test_compute_ipw_att(self):
        att_ipw = compute_ipw_att(self.A, self.Y, self.ps)
        self.assertAlmostEqual(att_ipw[EFFECT], self.true_att, delta=0.01)


class TestComputeIPW_RR(TestEffectBase):
    """Checks if IPW can recover the true RR in a well-behaved simulation."""

    def test_compute_ipw_rr(self):
        rr_ipw = compute_ipw_risk_ratio(self.A, self.Y, self.ps)
        self.assertAlmostEqual(rr_ipw[EFFECT], self.true_rr, delta=0.01)


# Run the unittests
if __name__ == "__main__":
    unittest.main()
