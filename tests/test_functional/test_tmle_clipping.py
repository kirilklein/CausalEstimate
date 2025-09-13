import unittest
import numpy as np

from CausalEstimate.estimators.functional.utils import (
    compute_clever_covariate_ate,
    compute_clever_covariate_att,
)
from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
)
from CausalEstimate.utils.constants import EFFECT


class TestTMLEClipping(unittest.TestCase):
    """
    Tests for the clip_percentile functionality in TMLE estimators.
    This tests the clipping of clever covariates to handle extreme propensity scores.
    """

    def setUp(self):
        """Create data with extreme propensity scores to test clipping."""
        np.random.seed(42)
        n = 100

        # Create extreme propensity scores (some very close to 0 and 1)
        self.ps_extreme = np.concatenate(
            [
                np.random.uniform(0.01, 0.05, size=20),  # Very low PS
                np.random.uniform(0.2, 0.8, size=60),  # Normal PS
                np.random.uniform(0.95, 0.99, size=20),  # Very high PS
            ]
        )

        # Randomly assign treatment based on PS (but ensure we have both groups)
        self.A = np.random.binomial(1, self.ps_extreme, size=n)
        # Ensure we have both treated and control subjects
        if self.A.sum() == 0:
            self.A[0] = 1
        if self.A.sum() == n:
            self.A[0] = 0

        # Create outcome data
        self.Y = np.random.binomial(1, 0.3 + 0.4 * self.A, size=n)

        # Create outcome predictions (clipped to avoid logit issues)
        self.Y1_hat = np.clip(np.random.uniform(0.5, 0.9, size=n), 0.01, 0.99)
        self.Y0_hat = np.clip(np.random.uniform(0.1, 0.5, size=n), 0.01, 0.99)
        self.Yhat = np.where(self.A == 1, self.Y1_hat, self.Y0_hat)

    def test_ate_clever_covariate_clipping_reduces_extreme_values(self):
        """Test that clipping reduces extreme values in ATE clever covariates."""
        # Compute clever covariate without clipping
        H_unclipped = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=1.0
        )

        # Compute clever covariate with clipping at 90th percentile
        H_clipped = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=0.9
        )

        # Check that maximum absolute value is reduced after clipping
        self.assertLess(np.abs(H_clipped).max(), np.abs(H_unclipped).max())

        # Check that extreme values are bounded
        self.assertLess(
            np.abs(H_clipped).max(), 1000
        )  # Should be much smaller than unclipped

        # Verify clipping was applied by checking that the 95th percentile is reduced
        self.assertLess(
            np.percentile(np.abs(H_clipped), 95), np.percentile(np.abs(H_unclipped), 95)
        )

    def test_att_clever_covariate_clipping_only_affects_controls(self):
        """Test that clipping for ATT only affects control group clever covariates."""
        # Compute clever covariate without clipping
        H_unclipped = compute_clever_covariate_att(
            self.A, self.ps_extreme, clip_percentile=1.0
        )

        # Compute clever covariate with clipping at 80th percentile
        H_clipped = compute_clever_covariate_att(
            self.A, self.ps_extreme, clip_percentile=0.8
        )

        treated_mask = self.A == 1
        control_mask = self.A == 0

        # Treated components should be identical (no clipping applied)
        if treated_mask.sum() > 0:
            np.testing.assert_array_equal(
                H_unclipped[treated_mask], H_clipped[treated_mask]
            )

        # Control components should be different (clipping applied)
        if control_mask.sum() > 0:
            # The maximum absolute value for controls should be reduced
            self.assertLessEqual(
                np.abs(H_clipped[control_mask]).max(),
                np.abs(H_unclipped[control_mask]).max(),
            )

    def test_clip_percentile_effect_on_ate_estimation(self):
        """Test that clipping affects ATE estimates by reducing influence of extreme weights."""
        # Estimate ATE without clipping
        ate_unclipped = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps_extreme,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            clip_percentile=1.0,
        )

        # Estimate ATE with clipping
        ate_clipped = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps_extreme,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            clip_percentile=0.9,
        )

        # The estimates should be different (clipping should reduce variance)
        self.assertNotEqual(ate_unclipped[EFFECT], ate_clipped[EFFECT])

        # Both estimates should be finite and reasonable
        self.assertTrue(np.isfinite(ate_clipped[EFFECT]))
        self.assertTrue(np.isfinite(ate_unclipped[EFFECT]))
        self.assertTrue(-2 <= ate_clipped[EFFECT] <= 2)  # Reasonable bounds

    def test_clip_percentile_effect_on_att_estimation(self):
        """Test that clipping affects ATT estimates."""
        # Estimate ATT without clipping
        att_unclipped = compute_tmle_att(
            self.A,
            self.Y,
            self.ps_extreme,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            clip_percentile=1.0,
        )

        # Estimate ATT with clipping
        att_clipped = compute_tmle_att(
            self.A,
            self.Y,
            self.ps_extreme,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            clip_percentile=0.85,
        )

        # The estimates should be different
        self.assertNotEqual(att_unclipped[EFFECT], att_clipped[EFFECT])

        # Both estimates should be finite and reasonable
        self.assertTrue(np.isfinite(att_clipped[EFFECT]))
        self.assertTrue(np.isfinite(att_unclipped[EFFECT]))
        self.assertTrue(-2 <= att_clipped[EFFECT] <= 2)

    def test_clip_percentile_effect_on_rr_estimation(self):
        """Test that clipping affects Risk Ratio estimates."""
        # Estimate RR without clipping
        rr_unclipped = compute_tmle_rr(
            self.A,
            self.Y,
            self.ps_extreme,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            clip_percentile=1.0,
        )

        # Estimate RR with clipping
        rr_clipped = compute_tmle_rr(
            self.A,
            self.Y,
            self.ps_extreme,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            clip_percentile=0.9,
        )

        # The estimates should be different
        self.assertNotEqual(rr_unclipped[EFFECT], rr_clipped[EFFECT])

        # Both estimates should be finite and positive
        self.assertTrue(np.isfinite(rr_clipped[EFFECT]))
        self.assertTrue(rr_clipped[EFFECT] > 0)

    def test_no_clipping_when_percentile_is_one(self):
        """Test that no clipping occurs when clip_percentile=1.0."""
        H_no_clip_1 = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=1.0
        )
        H_no_clip_2 = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=1.0
        )

        # Should be identical
        np.testing.assert_array_equal(H_no_clip_1, H_no_clip_2)

        # Should equal the theoretical unclipped values
        expected_H = self.A / self.ps_extreme - (1 - self.A) / (1 - self.ps_extreme)
        np.testing.assert_array_almost_equal(H_no_clip_1, expected_H)

    def test_extreme_clipping_percentiles(self):
        """Test behavior with very low clip percentiles."""
        # Test with very aggressive clipping (50th percentile)
        H_aggressive = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=0.5
        )

        # Should be heavily clipped but still finite
        self.assertTrue(np.all(np.isfinite(H_aggressive)))
        self.assertLess(np.abs(H_aggressive).max(), 100)

        # Test with very conservative clipping (10th percentile)
        H_conservative = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=0.1
        )

        # Should be even more heavily clipped
        self.assertLess(np.abs(H_conservative).max(), np.abs(H_aggressive).max())

    def test_clipping_with_edge_case_propensity_scores(self):
        """Test clipping behavior with edge case propensity scores."""
        # Create very extreme propensity scores
        ps_edge = np.array([0.001, 0.999, 0.5, 0.5])
        A_edge = np.array([1, 0, 1, 0])

        # This should not crash and should produce finite results
        H_clipped = compute_clever_covariate_ate(A_edge, ps_edge, clip_percentile=0.8)

        self.assertTrue(np.all(np.isfinite(H_clipped)))
        # After clipping, values should be much more reasonable
        self.assertLess(np.abs(H_clipped).max(), 1000)

    def test_clipping_preserves_clever_covariate_properties(self):
        """Test that clipping preserves important properties of clever covariates."""
        H_clipped = compute_clever_covariate_ate(
            self.A, self.ps_extreme, clip_percentile=0.9
        )

        # Should still be finite
        self.assertTrue(np.all(np.isfinite(H_clipped)))

        # Should have the same length as input
        self.assertEqual(len(H_clipped), len(self.A))

        # For ATT, test similar properties
        H_att_clipped = compute_clever_covariate_att(
            self.A, self.ps_extreme, clip_percentile=0.9
        )

        self.assertTrue(np.all(np.isfinite(H_att_clipped)))
        self.assertEqual(len(H_att_clipped), len(self.A))


if __name__ == "__main__":
    unittest.main()
