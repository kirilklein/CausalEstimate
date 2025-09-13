import unittest
import numpy as np

from CausalEstimate.estimators.functional.utils import (
    compute_clever_covariate_ate,
    compute_clever_covariate_att,
    compute_initial_effect,
    estimate_fluctuation_parameter,
    _compute_epsilon_one_step,
)
from CausalEstimate.utils.constants import (
    INITIAL_EFFECT,
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
    INITIAL_EFFECT_treated,
)


class TestTMLEUtils(unittest.TestCase):
    """Unit tests for the TMLE utility functions."""

    def setUp(self):
        """Set up a simple, fixed dataset for consistent testing."""
        self.A = np.array([1, 1, 0, 0])
        self.Y = np.array([1, 0, 1, 1])
        self.ps = np.array([0.8, 0.6, 0.4, 0.2])
        self.Yhat = np.array([0.7, 0.6, 0.5, 0.4])
        self.Y1_hat = np.array([0.75, 0.65, 0.55, 0.45])
        self.Y0_hat = np.array([0.65, 0.55, 0.45, 0.35])
        self.Q_star_1 = self.Y1_hat + 0.05
        self.Q_star_0 = self.Y0_hat - 0.05

    def test_compute_clever_covariate_ate_no_clipping(self):
        """Test ATE clever covariate calculation without clipping."""
        H = compute_clever_covariate_ate(self.A, self.ps, clip_percentile=1.0)
        expected = np.array([1 / 0.8, 1 / 0.6, -1 / (1 - 0.4), -1 / (1 - 0.2)])
        np.testing.assert_allclose(H, expected)

    def test_compute_clever_covariate_ate_with_clipping(self):
        """Test ATE clever covariate calculation with clipping."""
        # Create extreme propensity scores
        ps_extreme = np.array([0.99, 0.01, 0.5, 0.5])
        A_extreme = np.array([1, 0, 1, 0])

        # Without clipping - should have extreme values
        H_unclipped = compute_clever_covariate_ate(
            A_extreme, ps_extreme, clip_percentile=1.0
        )

        # With clipping - should have reduced extreme values
        H_clipped = compute_clever_covariate_ate(
            A_extreme, ps_extreme, clip_percentile=0.8
        )

        # Clipped version should have smaller maximum absolute value
        self.assertLess(np.abs(H_clipped).max(), np.abs(H_unclipped).max())

    def test_compute_clever_covariate_att_no_clipping(self):
        """Test ATT clever covariate calculation without clipping."""
        p_treated = 0.5

        H = compute_clever_covariate_att(self.A, self.ps, clip_percentile=1.0)
        expected = np.array(
            [
                1 / p_treated,
                1 / p_treated,
                -self.ps[2] / (p_treated * (1 - self.ps[2])),
                -self.ps[3] / (p_treated * (1 - self.ps[3])),
            ]
        )
        np.testing.assert_allclose(H, expected)

    def test_compute_clever_covariate_att_with_clipping(self):
        """Test ATT clever covariate calculation with clipping."""
        # Create extreme propensity scores
        ps_extreme = np.array([0.99, 0.99, 0.01, 0.01])
        A_extreme = np.array([1, 1, 0, 0])

        # Without clipping - should have extreme values for controls
        H_unclipped = compute_clever_covariate_att(
            A_extreme, ps_extreme, clip_percentile=1.0
        )

        # With clipping - should have reduced extreme values for controls
        H_clipped = compute_clever_covariate_att(
            A_extreme, ps_extreme, clip_percentile=0.8
        )

        # Treated components should be the same (A=1 positions)
        treated_mask = A_extreme == 1
        np.testing.assert_array_equal(
            H_unclipped[treated_mask], H_clipped[treated_mask]
        )

        # Control components should be clipped (A=0 positions)
        control_mask = A_extreme == 0
        if control_mask.sum() > 0:
            self.assertLessEqual(
                np.abs(H_clipped[control_mask]).max(),
                np.abs(H_unclipped[control_mask]).max(),
            )

    def test_clever_covariate_att_no_treated(self):
        """Test ATT clever covariate calculation when there are no treated subjects."""
        A_no_treated = np.array([0, 0, 0, 0])
        with self.assertWarns(RuntimeWarning):
            H = compute_clever_covariate_att(A_no_treated, self.ps, clip_percentile=1.0)
            # Should return an array of zeros
            np.testing.assert_allclose(H, np.zeros(4))

    def test_clever_covariate_extreme_value_warning(self):
        """Test that a warning is issued for extreme H values."""
        # Create propensity scores very close to 0 and 1 to generate large H
        ps_extreme = np.array([0.9999, 0.0001])
        A_extreme = np.array([0, 1])
        with self.assertWarns(RuntimeWarning):
            compute_clever_covariate_ate(A_extreme, ps_extreme, clip_percentile=1.0)

    def test_clipping_percentile_bounds(self):
        """Test that clipping percentile parameter is properly bounded."""
        # Test with valid clip_percentile values
        for clip_pct in [0.1, 0.5, 0.9, 1.0]:
            H = compute_clever_covariate_ate(self.A, self.ps, clip_percentile=clip_pct)
            self.assertTrue(np.all(np.isfinite(H)))

        # Test ATT as well
        for clip_pct in [0.1, 0.5, 0.9, 1.0]:
            H = compute_clever_covariate_att(self.A, self.ps, clip_percentile=clip_pct)
            self.assertTrue(np.all(np.isfinite(H)))

    def test_compute_initial_effect(self):
        """Test calculation of initial effect and adjustments."""
        results = compute_initial_effect(
            self.Y1_hat, self.Y0_hat, self.Q_star_1, self.Q_star_0, rr=False
        )

        expected_initial_effect = self.Y1_hat.mean() - self.Y0_hat.mean()
        expected_adj_1 = 0.05
        expected_adj_0 = -0.05

        self.assertAlmostEqual(results[INITIAL_EFFECT], expected_initial_effect)
        self.assertAlmostEqual(results[ADJUSTMENT_treated], expected_adj_1)
        self.assertAlmostEqual(results[ADJUSTMENT_untreated], expected_adj_0)
        self.assertIn(INITIAL_EFFECT_treated, results)

    def test_compute_initial_effect_rr_zero_denominator(self):
        """Test risk ratio calculation when the denominator is zero."""
        Y0_hat_zero = np.array([0.0, 0.0, 0.0])
        Y1_hat_one = np.array([1.0, 1.0, 1.0])

        with self.assertWarns(RuntimeWarning):
            results = compute_initial_effect(
                Y1_hat_one, Y0_hat_zero, Y1_hat_one, Y0_hat_zero, rr=True
            )
            self.assertEqual(results[INITIAL_EFFECT], np.inf)

    def test_estimate_fluctuation_parameter_stable_case(self):
        """Test epsilon estimation in a normal, stable case."""
        # In this case, H is not perfectly correlated with the residual
        H = np.array([-1, 1, -1, 1])
        Y = np.array([0, 1, 1, 0])  # Mixed residuals
        Yhat = np.full(4, 0.5)

        epsilon = estimate_fluctuation_parameter(H, Y, Yhat)
        # We expect a small, finite epsilon and no warnings
        self.assertTrue(np.isfinite(epsilon))
        self.assertLess(np.abs(epsilon), 1.0)

    def test_estimate_fluctuation_parameter_fallback_on_separation(self):
        """Test that epsilon estimation falls back to one-step on perfect separation."""
        # Create a perfect separation scenario: Y=1 for H>0, Y=0 for H<0
        # The iterative MLE will try to drive epsilon to infinity.
        H = np.array([-2, -1, 1, 2])
        Y = np.array([0, 0, 1, 1])
        Yhat = np.full(4, 0.5)

        # Manually calculate the one-step estimate that we expect as the fallback
        expected_one_step = _compute_epsilon_one_step(H, Y, Yhat)
        self.assertTrue(np.isfinite(expected_one_step))

        # Check that the function issues a warning and returns the one-step value
        with self.assertWarns(RuntimeWarning) as cm:
            epsilon = estimate_fluctuation_parameter(H, Y, Yhat)

        self.assertIn("Falling back to the one-step estimate", str(cm.warning))
        self.assertAlmostEqual(epsilon, expected_one_step)


if __name__ == "__main__":
    unittest.main()
