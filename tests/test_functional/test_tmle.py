import unittest
import warnings
from typing import List

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
    estimate_fluctuation_parameter,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
    estimate_fluctuation_parameter_att,
)
from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated
from tests.helpers.setup import TestEffectBase


class TestTMLEFunctions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter(self):
        epsilon = estimate_fluctuation_parameter(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATT_Functions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter_att(self):
        epsilon = estimate_fluctuation_parameter_att(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATE_base(TestEffectBase):
    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_PS_misspecified(TestTMLE_ATE_base):
    alpha = [0.1, 0.2, -0.3, 3]


class TestTMLE_OutcomeModel_misspecified(TestTMLE_ATE_base):
    beta = [0.5, 0.8, -0.6, 0.3, 3]


class TestTMLE_PS_misspecified_and_OutcomeModel_misspecified(TestTMLE_ATE_base):
    alpha = [0.1, 0.2, -0.3, 5]
    beta = [0.5, 0.8, -0.6, 0.3, 5]

    # extreme misspecification
    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertNotAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.1)


class TestTMLE_RR(TestEffectBase):
    def test_compute_tmle_rr(self):
        rr_tmle = compute_tmle_rr(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(rr_tmle[EFFECT], self.true_rr, delta=1)


class TestTMLE_RR_PS_misspecified(TestTMLE_RR):
    alpha = [0.1, 0.2, -0.3, 5]


class TestTMLE_RR_OutcomeModel_misspecified(TestTMLE_RR):
    beta = [0.5, 0.8, -0.6, 0.3, 5]


class TestTMLE_RR_PS_misspecified_and_OutcomeModel_misspecified(TestEffectBase):
    alpha = [0.1, 0.2, -0.3, 5]
    beta = [0.5, 0.8, -0.6, 0.3, 5]

    # extreme misspecification
    def test_compute_tmle_rr(self):
        rr_tmle = compute_tmle_rr(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertNotAlmostEqual(rr_tmle[EFFECT], self.true_rr, delta=0.1)


class TestTMLE_ATT(TestEffectBase):
    def test_compute_tmle_att(self):
        att_tmle = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(att_tmle[EFFECT], self.true_att, delta=0.02)


class TestTMLE_ATT_PS_misspecified(TestTMLE_ATT):
    alpha = [0.1, 0.2, -0.3, 5]


class TestTMLE_ATT_OutcomeModel_misspecified(TestTMLE_ATT):
    beta = [0.9, 0.8, -0.6, 0.3, 5, 1]


class TestTMLE_ATT_PS_misspecified_and_OutcomeModel_misspecified(TestTMLE_ATT):
    alpha = [0.1, 0.2, -0.3, 5]
    beta = [0.5, 0.8, -0.6, 0.3, 5]

    # extreme misspecification
    def test_compute_tmle_att(self):
        att_tmle = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertNotAlmostEqual(att_tmle[EFFECT], self.true_att, delta=0.1)


class TestTMLE_ATT_bounded(TestEffectBase):
    def test_att_is_bounded(self):
        att_tmle = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertLessEqual(att_tmle[EFFECT], 1)
        self.assertGreaterEqual(att_tmle[EFFECT], -1)


class TestTMLE_ATE_stabilized(TestEffectBase):
    """Checks if the stabilized TMLE ATE can recover the true effect."""

    def test_compute_tmle_ate_stabilized(self):
        ate_tmle = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_ATT_stabilized(TestEffectBase):
    """
    Checks if the stabilized TMLE ATT can recover the true effect.
    NOTE: This assumes a `stabilized` flag has been added to `compute_tmle_att`
    in the same way as `compute_tmle_ate`.
    """

    def test_compute_tmle_att_stabilized(self):
        att_tmle = compute_tmle_att(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )
        self.assertAlmostEqual(att_tmle[EFFECT], self.true_att, delta=0.02)


class TestTMLEStabilizationBenefit(TestEffectBase):
    """
    Demonstrates that stabilization reduces variance for the TMLE estimator
    by bootstrapping from a single, high-variance data simulation.
    """

    # Override alpha from TestEffectBase to create a high-variance scenario
    alpha: List[float] = [0.5, -2.5, 3.0, 0]
    # Use a larger sample for a more stable bootstrap base
    n: int = 5000

    def _get_bootstrap_standard_error(
        self, n_replicates: int, stabilized: bool
    ) -> float:
        """
        Calculates the standard error of the TMLE ATE estimate via bootstrap.
        It relies on the full data created by TestEffectBase.setUpClass.
        """
        rng = np.random.default_rng(self.seed)
        n_obs = len(self.A)
        bootstrap_ates = []

        for _ in range(n_replicates):
            # Create a bootstrap sample by drawing indices with replacement
            indices = rng.choice(n_obs, size=n_obs, replace=True)

            # Resample all necessary arrays for TMLE
            A_boot = self.A[indices]
            Y_boot = self.Y[indices]
            ps_boot = self.ps[indices]
            Y0_hat_boot = self.Y0_hat[indices]
            Y1_hat_boot = self.Y1_hat[indices]
            Yhat_boot = self.Yhat[indices]

            # Calculate the ATE on the resampled data using the actual TMLE function
            ate_boot = compute_tmle_ate(
                A_boot,
                Y_boot,
                ps_boot,
                Y0_hat_boot,
                Y1_hat_boot,
                Yhat_boot,
                stabilized=stabilized,
            )

            if not np.isnan(ate_boot[EFFECT]):
                bootstrap_ates.append(ate_boot[EFFECT])

        # The standard deviation of the bootstrap estimates is our standard error
        return np.std(bootstrap_ates)

    def test_stabilization_reduces_bootstrap_variance(self):
        """
        Asserts that the bootstrap standard error is smaller for the stabilized TMLE estimator.
        """
        n_replicates = 100  # Keep lower for speed, increase for precision

        # --- Estimate Standard Error for both estimators ---
        se_unstabilized = self._get_bootstrap_standard_error(
            n_replicates=n_replicates, stabilized=False
        )
        se_stabilized = self._get_bootstrap_standard_error(
            n_replicates=n_replicates, stabilized=True
        )

        print(
            f"\n[TMLE Stabilization Benefit] Unstabilized ATE SE: {se_unstabilized:.4f}"
        )
        print(f"[TMLE Stabilization Benefit] Stabilized ATE SE:   {se_stabilized:.4f}")

        # --- The Definitive Assertion ---
        self.assertLess(
            se_stabilized,
            se_unstabilized,
            "Stabilized TMLE should yield a lower bootstrap standard error, indicating reduced variance.",
        )


### Additional robusness tests ###


class TestTMLEEdgeCases(TestEffectBase):
    """Test TMLE behavior in edge cases and boundary conditions"""

    def test_extreme_propensity_scores(self):
        """Test TMLE with propensity scores very close to 0 and 1"""
        # Create extreme propensity scores
        ps_extreme = np.copy(self.ps)
        ps_extreme[:100] = 1e-8  # Very close to 0
        ps_extreme[-100:] = 1 - 1e-8  # Very close to 1

        # Test ATE - should generate warnings but not crash
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ate_result = compute_tmle_ate(
                self.A, self.Y, ps_extreme, self.Y0_hat, self.Y1_hat, self.Yhat
            )
            # Should generate warnings about extreme values
            self.assertTrue(len(w) > 0)
            self.assertTrue(
                any("Extremely large values" in str(warning.message) for warning in w)
            )

        # Result should still be finite
        self.assertTrue(np.isfinite(ate_result[EFFECT]))

        # Test ATT with same extreme propensity scores
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            att_result = compute_tmle_att(
                self.A, self.Y, ps_extreme, self.Y0_hat, self.Y1_hat, self.Yhat
            )
            self.assertTrue(np.isfinite(att_result[EFFECT]))

    def test_all_treated_or_all_control(self):
        """Test TMLE when all subjects are treated or all are controls"""
        n = len(self.A)

        # All treated case
        A_all_treated = np.ones(n, dtype=int)
        ps_all_treated = np.full(n, 0.99)  # High but not exactly 1

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            ate_all_treated = compute_tmle_ate(
                A_all_treated,
                self.Y,
                ps_all_treated,
                self.Y0_hat,
                self.Y1_hat,
                self.Yhat,
            )
            # Should handle gracefully
            self.assertTrue(np.isfinite(ate_all_treated[EFFECT]))

        # All control case
        A_all_control = np.zeros(n, dtype=int)
        ps_all_control = np.full(n, 0.01)  # Low but not exactly 0

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ate_all_control = compute_tmle_ate(
                A_all_control,
                self.Y,
                ps_all_control,
                self.Y0_hat,
                self.Y1_hat,
                self.Yhat,
            )
            self.assertTrue(np.isfinite(ate_all_control[EFFECT]))

    def test_outcome_predictions_at_boundaries(self):
        """Test TMLE with outcome predictions at 0 and 1 boundaries"""
        # Create boundary outcome predictions
        Y0_hat_boundary = np.copy(self.Y0_hat)
        Y1_hat_boundary = np.copy(self.Y1_hat)
        Yhat_boundary = np.copy(self.Yhat)

        # Set some predictions to boundaries (but not exactly due to clipping)
        Y0_hat_boundary[:50] = 1e-6  # Near 0
        Y1_hat_boundary[:50] = 1 - 1e-6  # Near 1
        Yhat_boundary[:50] = 1e-6

        # Should handle without numerical issues
        ate_boundary = compute_tmle_ate(
            self.A, self.Y, self.ps, Y0_hat_boundary, Y1_hat_boundary, Yhat_boundary
        )
        self.assertTrue(np.isfinite(ate_boundary[EFFECT]))

        # Test RR with boundary values
        rr_boundary = compute_tmle_rr(
            self.A, self.Y, self.ps, Y0_hat_boundary, Y1_hat_boundary, Yhat_boundary
        )
        self.assertTrue(
            np.isfinite(rr_boundary[EFFECT]) or rr_boundary[EFFECT] == np.inf
        )


class TestTMLERiskRatioSpecialCases(TestEffectBase):
    """Test specific edge cases for Risk Ratio estimation"""

    def test_zero_control_outcome_risk_ratio(self):
        """Test RR when control group has zero expected outcome"""
        # Create scenario where Y0_hat is very close to 0
        Y0_hat_zero = np.full_like(self.Y0_hat, 1e-10)
        Y1_hat_nonzero = np.full_like(self.Y1_hat, 0.5)
        Yhat_mixed = self.A * Y1_hat_nonzero + (1 - self.A) * Y0_hat_zero

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr_result = compute_tmle_rr(
                self.A, self.Y, self.ps, Y0_hat_zero, Y1_hat_nonzero, Yhat_mixed
            )

            # Should warn about zero denominator and return inf
            self.assertTrue(
                any("Mean of Q_star_0 is 0" in str(warning.message) for warning in w)
            )
            self.assertEqual(rr_result[EFFECT], np.inf)


class TestTMLENumericalStability(TestEffectBase):
    """Test numerical stability and precision of TMLE calculations"""

    def test_small_sample_stability(self):
        """Test TMLE with very small sample sizes"""
        small_n = 20
        indices = np.random.choice(len(self.A), size=small_n, replace=False)

        A_small = self.A[indices]
        Y_small = self.Y[indices]
        ps_small = self.ps[indices]
        Y0_hat_small = self.Y0_hat[indices]
        Y1_hat_small = self.Y1_hat[indices]
        Yhat_small = self.Yhat[indices]

        # Should not crash with small samples
        ate_small = compute_tmle_ate(
            A_small, Y_small, ps_small, Y0_hat_small, Y1_hat_small, Yhat_small
        )
        att_small = compute_tmle_att(
            A_small, Y_small, ps_small, Y0_hat_small, Y1_hat_small, Yhat_small
        )

        self.assertTrue(np.isfinite(ate_small[EFFECT]))
        self.assertTrue(np.isfinite(att_small[EFFECT]))

    def test_large_sample_consistency(self):
        """Test that TMLE estimates remain stable with large samples"""
        # Use a larger sample by repeating the data
        repetitions = 5
        large_A = np.tile(self.A, repetitions)
        large_Y = np.tile(self.Y, repetitions)
        large_ps = np.tile(self.ps, repetitions)
        large_Y0_hat = np.tile(self.Y0_hat, repetitions)
        large_Y1_hat = np.tile(self.Y1_hat, repetitions)
        large_Yhat = np.tile(self.Yhat, repetitions)

        ate_large = compute_tmle_ate(
            large_A, large_Y, large_ps, large_Y0_hat, large_Y1_hat, large_Yhat
        )
        ate_original = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )

        # Results should be very similar
        self.assertAlmostEqual(ate_large[EFFECT], ate_original[EFFECT], delta=0.001)

    def test_repeated_calculations_consistency(self):
        """Test that repeated calculations give identical results"""
        results = []
        for _ in range(5):
            ate_result = compute_tmle_ate(
                self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
            )
            results.append(ate_result[EFFECT])

        # All results should be identical
        for result in results[1:]:
            self.assertEqual(results[0], result)


class TestTMLEInputValidation(TestEffectBase):
    """Test TMLE behavior with invalid or edge case inputs"""

    def test_mismatched_array_lengths(self):
        """Test TMLE with arrays of different lengths"""
        A_short = self.A[:-100]

        with self.assertRaises((ValueError, IndexError)):
            compute_tmle_ate(
                A_short, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
            )

    def test_nan_values_in_inputs(self):
        """Test TMLE behavior when inputs contain NaN values"""
        ps_with_nan = np.copy(self.ps)
        ps_with_nan[0] = np.nan

        with self.assertRaises(Exception):
            compute_tmle_ate(
                self.A, self.Y, ps_with_nan, self.Y0_hat, self.Y1_hat, self.Yhat
            )


class TestTMLEReturnStructure(TestEffectBase):
    """Test the structure and content of TMLE return values"""

    def test_ate_return_keys(self):
        """Test that ATE results contain all expected keys"""
        ate_result = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )

        expected_keys = {
            EFFECT,
            EFFECT_treated,
            EFFECT_untreated,
            "initial_effect",
            "initial_effect_1",
            "initial_effect_0",
            "adjustment_1",
            "adjustment_0",
        }

        self.assertTrue(expected_keys.issubset(set(ate_result.keys())))

        # All values should be numeric
        for key, value in ate_result.items():
            self.assertTrue(
                np.isfinite(value), f"Non-finite value for key {key}: {value}"
            )

    def test_att_return_keys(self):
        """Test that ATT results contain all expected keys"""
        att_result = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )

        expected_keys = {
            EFFECT,
            EFFECT_treated,
            EFFECT_untreated,
            "initial_effect",
            "initial_effect_1",
            "initial_effect_0",
            "adjustment_1",
            "adjustment_0",
        }

        self.assertTrue(expected_keys.issubset(set(att_result.keys())))

        # All values should be numeric
        for key, value in att_result.items():
            self.assertTrue(
                np.isfinite(value), f"Non-finite value for key {key}: {value}"
            )

    def test_rr_return_keys(self):
        """Test that RR results contain all expected keys"""
        rr_result = compute_tmle_rr(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )

        expected_keys = {
            EFFECT,
            EFFECT_treated,
            EFFECT_untreated,
            "initial_effect",
            "initial_effect_1",
            "initial_effect_0",
            "adjustment_1",
            "adjustment_0",
        }

        self.assertTrue(expected_keys.issubset(set(rr_result.keys())))

        # Effect should be positive for RR
        self.assertGreaterEqual(rr_result[EFFECT], 0)


class TestTMLEStabilizedVsUnstabilized(TestEffectBase):
    """Comprehensive comparison of stabilized vs unstabilized TMLE"""

    def test_stabilized_unstabilized_ate_comparison(self):
        """Compare stabilized vs unstabilized ATE estimates"""
        ate_unstabilized = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=False,
        )
        ate_stabilized = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )

        # Both should be finite
        self.assertTrue(np.isfinite(ate_unstabilized[EFFECT]))
        self.assertTrue(np.isfinite(ate_stabilized[EFFECT]))

        # Effects should be reasonably close (within 20% for well-behaved data)
        relative_diff = abs(ate_stabilized[EFFECT] - ate_unstabilized[EFFECT]) / abs(
            ate_unstabilized[EFFECT]
        )
        self.assertLess(
            relative_diff,
            0.2,
            f"Stabilized and unstabilized estimates differ too much: {relative_diff}",
        )

    def test_stabilized_unstabilized_att_comparison(self):
        """Compare stabilized vs unstabilized ATT estimates"""
        att_unstabilized = compute_tmle_att(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=False,
        )
        att_stabilized = compute_tmle_att(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )

        # Both should be finite
        self.assertTrue(np.isfinite(att_unstabilized[EFFECT]))
        self.assertTrue(np.isfinite(att_stabilized[EFFECT]))

        # Effects should be reasonably close
        relative_diff = abs(att_stabilized[EFFECT] - att_unstabilized[EFFECT]) / abs(
            att_unstabilized[EFFECT]
        )
        self.assertLess(
            relative_diff,
            0.2,
            f"Stabilized and unstabilized ATT estimates differ too much: {relative_diff}",
        )


class TestTMLEFluctuationParameter(TestEffectBase):
    """Test the fluctuation parameter estimation in detail"""

    def test_fluctuation_parameter_properties(self):
        """Test properties of the estimated fluctuation parameter"""
        epsilon_ate = estimate_fluctuation_parameter(self.A, self.Y, self.ps, self.Yhat)
        epsilon_att = estimate_fluctuation_parameter_att(
            self.A, self.Y, self.ps, self.Yhat
        )

        # Should be finite numbers
        self.assertTrue(np.isfinite(epsilon_ate))
        self.assertTrue(np.isfinite(epsilon_att))

        # Should be reasonably bounded for well-behaved data
        self.assertLess(
            abs(epsilon_ate), 10, "Fluctuation parameter suspiciously large"
        )
        self.assertLess(
            abs(epsilon_att), 10, "ATT fluctuation parameter suspiciously large"
        )

    def test_fluctuation_parameter_stabilized_vs_unstabilized(self):
        """Test fluctuation parameter with and without stabilization"""
        epsilon_unstabilized = estimate_fluctuation_parameter(
            self.A, self.Y, self.ps, self.Yhat, stabilized=False
        )
        epsilon_stabilized = estimate_fluctuation_parameter(
            self.A, self.Y, self.ps, self.Yhat, stabilized=True
        )

        # Both should be finite
        self.assertTrue(np.isfinite(epsilon_unstabilized))
        self.assertTrue(np.isfinite(epsilon_stabilized))

        # They may differ but should both be reasonable
        self.assertLess(abs(epsilon_unstabilized), 20)
        self.assertLess(abs(epsilon_stabilized), 20)


if __name__ == "__main__":
    unittest.main()
