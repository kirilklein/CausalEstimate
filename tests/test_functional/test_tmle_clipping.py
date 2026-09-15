import unittest
import numpy as np

from CausalEstimate.estimators.functional.utils import compute_arm_weights
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
    This tests the clipping of the targeting weights to handle extreme
    propensity scores.
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

    def test_ate_clipping_reduces_extreme_weights(self):
        """Clipping pulls in the upper tail of the ATE targeting weights."""
        w1_unclipped, w0_unclipped = compute_arm_weights(
            self.A, self.ps_extreme, "ATE", clip_percentile=1.0
        )
        w1_clipped, w0_clipped = compute_arm_weights(
            self.A, self.ps_extreme, "ATE", clip_percentile=0.9
        )

        H_unclipped = w1_unclipped - w0_unclipped
        H_clipped = w1_clipped - w0_clipped

        # The largest weight is pulled in.
        self.assertLess(np.abs(H_clipped).max(), np.abs(H_unclipped).max())
        self.assertLess(np.abs(H_clipped).max(), 1000)

        # The whole upper tail moves, not just the single largest weight.
        self.assertLess(
            np.percentile(np.abs(H_clipped), 95), np.percentile(np.abs(H_unclipped), 95)
        )

    def test_att_clipping_only_affects_controls(self):
        """
        The ATT treated weight is the constant 1/p_treated, so only the
        control arm has a tail for clipping to act on.
        """
        w1_unclipped, w0_unclipped = compute_arm_weights(
            self.A, self.ps_extreme, "ATT", clip_percentile=1.0
        )
        w1_clipped, w0_clipped = compute_arm_weights(
            self.A, self.ps_extreme, "ATT", clip_percentile=0.8
        )

        np.testing.assert_array_equal(w1_unclipped, w1_clipped)
        self.assertLessEqual(w0_clipped.max(), w0_unclipped.max())

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
        """
        At clip_percentile=1 the weights are the textbook covariate, which is
        the contract the ATE/ATT variance step depends on.
        """
        w1, w0 = compute_arm_weights(self.A, self.ps_extreme, "ATE", 1.0)

        expected_H = self.A / self.ps_extreme - (1 - self.A) / (1 - self.ps_extreme)
        np.testing.assert_array_almost_equal(w1 - w0, expected_H)

    def test_lower_clip_percentile_clips_harder(self):
        """A lower quantile is strictly more aggressive, and stays finite."""
        w1_mid, w0_mid = compute_arm_weights(self.A, self.ps_extreme, "ATE", 0.5)
        w1_low, w0_low = compute_arm_weights(self.A, self.ps_extreme, "ATE", 0.1)

        H_mid = w1_mid - w0_mid
        H_low = w1_low - w0_low

        self.assertTrue(np.all(np.isfinite(H_mid)))
        self.assertLess(np.abs(H_mid).max(), 100)
        self.assertLess(np.abs(H_low).max(), np.abs(H_mid).max())

    def test_clipping_with_edge_case_propensity_scores(self):
        """
        Propensity scores at the boundary must not crash or leak non-finite
        weights, and the result keeps the shape of the input.
        """
        ps_edge = np.array([0.001, 0.999, 0.5, 0.5])
        A_edge = np.array([1, 0, 1, 0])

        for effect_type in ("ATE", "ATT"):
            with self.subTest(effect_type=effect_type):
                w1, w0 = compute_arm_weights(A_edge, ps_edge, effect_type, 0.8)
                w1_raw, w0_raw = compute_arm_weights(A_edge, ps_edge, effect_type, 1.0)
                H = w1 - w0
                self.assertEqual(len(H), len(A_edge))
                self.assertTrue(np.all(np.isfinite(H)))
                # The absolute size of the surviving weight is estimand
                # specific, so assert the property that is not: clipping
                # brought the worst weight down.
                self.assertLess(np.abs(H).max(), np.abs(w1_raw - w0_raw).max())


if __name__ == "__main__":
    unittest.main()
