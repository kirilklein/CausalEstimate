import unittest
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_weights,
    compute_ipw_ate,
    compute_ipw_att,
)
import numpy as np
from CausalEstimate.utils.constants import EFFECT


class TestClipPercentileFunctionality(unittest.TestCase):
    """
    Tests for the clip_percentile functionality in IPW estimators.
    """

    @classmethod
    def setUpClass(cls):
        # Same setup as before to ensure we have extreme values
        rng = np.random.default_rng(42)
        n = 100
        cls.A = rng.choice([0, 1], size=n, p=[0.6, 0.4])
        cls.Y = rng.binomial(1, 0.3 + 0.2 * cls.A)
        cls.ps_extreme = np.concatenate(
            [
                rng.uniform(0.01, 0.05, size=10),
                rng.uniform(0.2, 0.8, size=80),
                rng.uniform(0.95, 0.99, size=10),
            ]
        )
        rng.shuffle(cls.ps_extreme)

    def test_ate_weight_clipping_is_correct(self):
        """Test that ATE clipping correctly caps weights at the specified percentile."""
        weights_unclipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=1.0
        )
        weights_clipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=0.9
        )

        treated_mask = self.A == 1
        control_mask = self.A == 0

        # IMPROVEMENT: Instead of just checking if the max is less, we check
        # that the new max is EXACTLY the 90th percentile threshold. This is a much stronger guarantee.
        if treated_mask.sum() > 0:
            threshold_treated = np.percentile(weights_unclipped[treated_mask], 90)
            self.assertTrue(np.all(weights_clipped[treated_mask] <= threshold_treated))
            # The new maximum value should be the threshold itself
            self.assertAlmostEqual(
                weights_clipped[treated_mask].max(), threshold_treated
            )

        if control_mask.sum() > 0:
            threshold_control = np.percentile(weights_unclipped[control_mask], 90)
            self.assertTrue(np.all(weights_clipped[control_mask] <= threshold_control))
            self.assertAlmostEqual(
                weights_clipped[control_mask].max(), threshold_control
            )

    def test_ate_weight_clipping_reduces_extreme_weights(self):
        """Test that clipping reduces extreme weights for ATE estimation."""
        # Compute weights without clipping
        weights_unclipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=1.0
        )

        # Compute weights with clipping at 90th percentile
        weights_clipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=0.9
        )

        # Check that maximum weight is reduced after clipping
        self.assertLess(weights_clipped.max(), weights_unclipped.max())

        # Check that 90th percentile is respected
        treated_mask = self.A == 1
        control_mask = self.A == 0

        if treated_mask.sum() > 0:
            treated_weights_unclipped = weights_unclipped[treated_mask]
            treated_weights_clipped = weights_clipped[treated_mask]
            threshold_treated = np.percentile(treated_weights_unclipped, 90)
            self.assertTrue(np.all(treated_weights_clipped <= threshold_treated))

        if control_mask.sum() > 0:
            control_weights_unclipped = weights_unclipped[control_mask]
            control_weights_clipped = weights_clipped[control_mask]
            threshold_control = np.percentile(control_weights_unclipped, 90)
            self.assertTrue(np.all(control_weights_clipped <= threshold_control))

    def test_att_weight_clipping_only_affects_controls(self):
        """Test that clipping for ATT only affects control group weights."""
        # Compute weights without clipping
        weights_unclipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATT", clip_percentile=1.0
        )

        # Compute weights with clipping at 80th percentile
        weights_clipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATT", clip_percentile=0.8
        )

        treated_mask = self.A == 1
        control_mask = self.A == 0

        # Treated weights should remain unchanged (always 1.0)
        if treated_mask.sum() > 0:
            np.testing.assert_array_equal(
                weights_unclipped[treated_mask], weights_clipped[treated_mask]
            )
            self.assertTrue(np.all(weights_clipped[treated_mask] == 1.0))

        # Control weights should be clipped
        if control_mask.sum() > 0:
            control_weights_unclipped = weights_unclipped[control_mask]
            control_weights_clipped = weights_clipped[control_mask]
            threshold = np.percentile(control_weights_unclipped, 80)
            self.assertTrue(np.all(control_weights_clipped <= threshold))

    def test_att_weight_clipping_only_affects_controls_correctly(self):
        """Test that clipping for ATT only affects control group weights and does so correctly."""
        weights_unclipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATT", clip_percentile=1.0
        )
        weights_clipped = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATT", clip_percentile=0.8
        )

        # This part of your test was already excellent and precise
        treated_mask = self.A == 1
        if treated_mask.sum() > 0:
            np.testing.assert_array_equal(
                weights_unclipped[treated_mask], weights_clipped[treated_mask]
            )
            self.assertTrue(np.all(weights_clipped[treated_mask] == 1.0))

        # IMPROVEMENT: Add a more precise check for the control group
        control_mask = self.A == 0
        if control_mask.sum() > 0:
            threshold = np.percentile(weights_unclipped[control_mask], 80)
            self.assertTrue(np.all(weights_clipped[control_mask] <= threshold))
            self.assertAlmostEqual(weights_clipped[control_mask].max(), threshold)

    def test_clip_percentile_effect_on_ate_estimation(self):
        """Test that clipping affects ATE estimates by reducing influence of extreme weights."""
        # Estimate ATE without clipping
        ate_unclipped = compute_ipw_ate(
            self.A, self.Y, self.ps_extreme, clip_percentile=1.0
        )

        # Estimate ATE with clipping
        ate_clipped = compute_ipw_ate(
            self.A, self.Y, self.ps_extreme, clip_percentile=0.9
        )

        # The estimates should be different (clipping should reduce variance)
        self.assertNotEqual(ate_unclipped[EFFECT], ate_clipped[EFFECT])

        # Both should still be reasonable values
        self.assertTrue(-1 <= ate_clipped[EFFECT] <= 1)
        self.assertTrue(-1 <= ate_unclipped[EFFECT] <= 1)

    def test_clip_percentile_effect_on_att_estimation(self):
        """Test that clipping affects ATT estimates."""
        # Estimate ATT without clipping
        att_unclipped = compute_ipw_att(
            self.A, self.Y, self.ps_extreme, clip_percentile=1.0
        )

        # Estimate ATT with clipping
        att_clipped = compute_ipw_att(
            self.A, self.Y, self.ps_extreme, clip_percentile=0.85
        )

        # The estimates should be different
        self.assertNotEqual(att_unclipped[EFFECT], att_clipped[EFFECT])

        # Both should still be reasonable values
        self.assertTrue(-1 <= att_clipped[EFFECT] <= 1)
        self.assertTrue(-1 <= att_unclipped[EFFECT] <= 1)

    def test_no_clipping_when_percentile_is_one(self):
        """Test that no clipping occurs when clip_percentile=1.0. (This test was already excellent)."""
        weights_no_clip = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=1.0
        )
        expected_weights = np.where(
            self.A == 1, 1 / self.ps_extreme, 1 / (1 - self.ps_extreme)
        )
        np.testing.assert_allclose(weights_no_clip, expected_weights, atol=1e-5)

    def test_extreme_clipping_percentiles(self):
        """Test behavior with very low clip percentiles."""
        # Test with very aggressive clipping (50th percentile)
        weights_aggressive = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=0.5
        )

        # All weights should be reasonable (not extremely large)
        self.assertTrue(np.all(weights_aggressive > 0))
        self.assertTrue(np.all(weights_aggressive < 100))  # Reasonable upper bound

        # Test that it doesn't crash with very small percentiles
        weights_very_aggressive = compute_ipw_weights(
            self.A, self.ps_extreme, weight_type="ATE", clip_percentile=0.1
        )
        self.assertTrue(np.all(weights_very_aggressive > 0))


class TestIPWWeightClippingDeterministic(unittest.TestCase):
    """
    Tests the `compute_ipw_weights` clipping with a small, deterministic
    dataset to verify exact numerical correctness.
    """

    @classmethod
    def setUpClass(cls):
        # Simple, predictable data for testing calculations
        cls.A = np.array([1, 1, 0, 0])
        cls.ps = np.array([0.8, 0.4, 0.5, 0.2])

    def test_ate_weights_clipping_exact_value(self):
        """Tests ATE weight clipping against a hand-calculated expected result."""
        # Unclipped treated weights: [1/0.8, 1/0.4] -> [1.25, 2.5]
        # Unclipped control weights: [1/0.5, 1/0.8] -> [2.0, 1.25]
        # Treated threshold (50th percentile): 1.875
        # Control threshold (50th percentile): 1.625
        expected_weights = np.array([1.25, 1.875, 1.625, 1.25])

        clipped_weights = compute_ipw_weights(
            self.A, self.ps, weight_type="ATE", clip_percentile=0.5
        )
        np.testing.assert_allclose(clipped_weights, expected_weights)

    def test_att_weights_clipping_exact_value(self):
        """Tests ATT weight clipping against a hand-calculated expected result."""
        # Unclipped treated weights: [1.0, 1.0]
        # Unclipped control weights: [0.5/0.5, 0.2/0.8] -> [1.0, 0.25]
        # Control threshold (50th percentile): 0.625
        # Treated weights are unchanged.
        expected_weights = np.array([1.0, 1.0, 0.625, 0.25])

        clipped_weights = compute_ipw_weights(
            self.A, self.ps, weight_type="ATT", clip_percentile=0.5
        )
        np.testing.assert_allclose(clipped_weights, expected_weights)


# Run the unittests
if __name__ == "__main__":
    unittest.main()
