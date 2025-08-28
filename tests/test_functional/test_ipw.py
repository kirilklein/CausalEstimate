import unittest
from typing import List

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

    def test_ipw_ate_stabilized(self):
        # This test now runs on data guaranteed to have both groups
        ate_stabilized = compute_ipw_ate(self.A, self.Y, self.ps, stabilized=True)
        self.assertIsInstance(ate_stabilized[EFFECT], float)
        self.assertTrue(-1 <= ate_stabilized[EFFECT] <= 1)

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

    def test_ipw_ate_stabilized(self):
        ate_stabilized = compute_ipw_ate(self.A, self.Y, self.ps, stabilized=True)
        self.assertIsInstance(ate_stabilized[EFFECT], float)
        self.assertTrue(
            -1 <= ate_stabilized[EFFECT] <= 1
        )  # Check ATE with stabilized weights

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

    def test_ate_stabilized_weights(self):
        weights = compute_ipw_weights(
            self.A, self.ps, weight_type="ATE", stabilized=True
        )
        expected = np.array(
            [
                self.pi / 0.8,
                self.pi / 0.4,
                (1 - self.pi) / (1 - 0.5),
                (1 - self.pi) / (1 - 0.2),
            ]
        )
        np.testing.assert_allclose(weights, expected)

    def test_att_stabilized_weights(self):
        weights = compute_ipw_weights(
            self.A, self.ps, weight_type="ATT", stabilized=True
        )
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


class TestComputeIPWATE_outcome_model_misspecified(TestComputeIPW_base):
    beta = [0.5, 0.8, -0.6, 0.3, 3]


class TestComputeIPWATE_ps_model_misspecified(TestComputeIPW_base):
    alpha = [0.1, 0.2, -0.3, 3]


class TestComputeIPWATE_both_models_misspecified(TestComputeIPW_base):
    beta = [0.5, 0.8, -0.6, 0.3, 3]
    alpha = [0.1, 0.2, -0.3, 3]

    def test_compute_ipw_ate(self):
        ate_ipw = compute_ipw_ate(self.A, self.Y, self.ps)
        self.assertNotAlmostEqual(ate_ipw[EFFECT], self.true_ate, delta=0.05)


class TestComputeIPW_ATT(TestEffectBase):
    """Checks if IPW can recover the true ATT in a well-behaved simulation."""

    def test_compute_ipw_att(self):
        att_ipw = compute_ipw_att(self.A, self.Y, self.ps)
        self.assertAlmostEqual(att_ipw[EFFECT], self.true_att, delta=0.1)


class TestComputeIPW_ATT_stabilized(TestEffectBase):
    """Checks if IPW can recover the true ATT in a well-behaved simulation."""

    def test_compute_ipw_att(self):
        att_ipw = compute_ipw_att(self.A, self.Y, self.ps, stabilized=True)
        self.assertAlmostEqual(att_ipw[EFFECT], self.true_att, delta=0.1)


class TestComputeIPW_RR(TestEffectBase):
    """Checks if IPW can recover the true RR in a well-behaved simulation."""

    def test_compute_ipw_rr(self):
        rr_ipw = compute_ipw_risk_ratio(self.A, self.Y, self.ps)
        self.assertAlmostEqual(rr_ipw[EFFECT], self.true_rr, delta=0.1)


class TestComputeIPW_RR_stabilized(TestEffectBase):
    """Checks if IPW can recover the true RR in a well-behaved simulation."""

    def test_compute_ipw_rr(self):
        rr_ipw = compute_ipw_risk_ratio(self.A, self.Y, self.ps, stabilized=True)
        self.assertAlmostEqual(rr_ipw[EFFECT], self.true_rr, delta=0.2)


class TestIPWStabilizationBenefit(TestEffectBase):
    """
    Demonstrates that stabilization reduces variance by bootstrapping from a
    single, high-variance data simulation.
    """

    # Override alpha from TestEffectBase to create a high-variance scenario
    alpha: List[float] = [0.5, -2.5, 3.0, 0]
    # Use a larger sample for a more stable bootstrap base
    n: int = 5000

    def _get_bootstrap_standard_error(
        self, n_replicates: int, stabilized: bool
    ) -> float:
        """
        Calculates the standard error of the ATE estimate via bootstrap.
        It relies on the self.A, self.Y, self.ps data created by TestEffectBase.
        """
        rng = np.random.default_rng(self.seed)
        n_obs = len(self.A)
        bootstrap_ates = []

        for _ in range(n_replicates):
            # Create a bootstrap sample by drawing indices with replacement
            indices = rng.choice(n_obs, size=n_obs, replace=True)
            A_boot, Y_boot, ps_boot = self.A[indices], self.Y[indices], self.ps[indices]

            # Calculate the ATE on the resampled data
            ate_boot = compute_ipw_ate(A_boot, Y_boot, ps_boot, stabilized=stabilized)

            if not np.isnan(ate_boot[EFFECT]):
                bootstrap_ates.append(ate_boot[EFFECT])

        # The standard deviation of the bootstrap estimates is our standard error
        return np.std(bootstrap_ates)

    def test_stabilization_reduces_bootstrap_variance(self):
        """
        Asserts that the bootstrap standard error is smaller for the stabilized estimator.
        """
        n_replicates = 200  # More replicates give a more stable SE estimate

        # --- Estimate Standard Error for both estimators ---
        se_unstabilized = self._get_bootstrap_standard_error(
            n_replicates=n_replicates, stabilized=False
        )
        se_stabilized = self._get_bootstrap_standard_error(
            n_replicates=n_replicates, stabilized=True
        )

        print(
            f"\n[Stabilization Benefit Test] Unstabilized ATE SE: {se_unstabilized:.4f}"
        )
        print(f"[Stabilization Benefit Test] Stabilized ATE SE:   {se_stabilized:.4f}")

        # --- The Definitive Assertion ---
        self.assertLess(
            se_stabilized,
            se_unstabilized,
            "Stabilized weights should yield a lower bootstrap standard error, indicating reduced variance.",
        )


# Run the unittests
if __name__ == "__main__":
    unittest.main()
