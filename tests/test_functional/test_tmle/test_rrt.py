"""
Tests for the TMLE estimator of the risk ratio in the treated.

The RRT reuses the ATT targeting step and differs only in how the two targeted
treated-arm means are combined, so most of these tests pin that relationship
rather than re-deriving the targeting machinery.
"""

import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle import compute_tmle_rr
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
    compute_tmle_rrt,
)
from CausalEstimate.simulation.binary_simulation import (
    compute_expected_outcome,
    simulate_binary_data,
)
from CausalEstimate.utils.constants import (
    ADJUSTMENT_untreated,
    CI95_LOWER,
    CI95_UPPER,
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
    INITIAL_EFFECT,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
    STD_ERR,
    TREATMENT_COL,
)
from tests.helpers.setup import TestEffectBase


class TestTMLERRT(TestEffectBase):
    """The TMLE RRT recovers the truth; subclassed per DGP."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        data = simulate_binary_data(
            cls.n, alpha=cls.alpha, beta=cls.beta, seed=cls.seed
        )
        treated = data[data[TREATMENT_COL] == 1]
        cls.true_rrt = compute_expected_outcome(
            treated, cls.beta, 1
        ) / compute_expected_outcome(treated, cls.beta, 0)

    def test_recovers_the_true_rrt(self):
        rrt = compute_tmle_rrt(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(rrt[EFFECT], self.true_rrt, delta=0.05)


class TestTMLERRTOutcomeModelMisspecified(TestTMLERRT):
    """A correct propensity model should rescue a wrong outcome model."""

    beta = [0.5, 0.8, -0.6, 0.3, 5]


class TestTMLERRTProperties(TestEffectBase):
    """
    Exact properties of the TMLE RRT. These hold on any data, so they run
    once rather than per DGP.
    """

    def _rrt(self):
        return compute_tmle_rrt(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)

    def test_arm_means_match_the_tmle_att(self):
        """
        Same targeting step, so only the combination step may differ. Since
        the ATT averages over the treated, so do the RRT's arm means.
        """
        rrt = self._rrt()
        att = compute_tmle_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(rrt[EFFECT_treated], att[EFFECT_treated], places=12)
        self.assertAlmostEqual(rrt[EFFECT_untreated], att[EFFECT_untreated], places=12)
        self.assertAlmostEqual(
            rrt[EFFECT], att[EFFECT_treated] / att[EFFECT_untreated], places=10
        )

    def test_targeted_treated_mean_equals_the_observed_treated_mean(self):
        """
        Solving the treated score equation forces the targeted treated mean onto
        the observed one, which is how the RRT is identified without a model
        of Y(1).
        """
        rrt = self._rrt()
        self.assertAlmostEqual(
            rrt[EFFECT_treated], float(self.Y[self.A == 1].mean()), places=8
        )

    def test_differs_from_the_marginal_rr(self):
        rr = compute_tmle_rr(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertNotAlmostEqual(self._rrt()[EFFECT], rr[EFFECT], places=6)

    def test_reports_a_positive_ci_bracketing_the_estimate(self):
        rrt = self._rrt()
        for key in (STD_ERR, CI95_LOWER, CI95_UPPER):
            self.assertTrue(np.isfinite(rrt[key]), key)
        self.assertGreater(rrt[CI95_LOWER], 0.0)
        self.assertLess(rrt[CI95_LOWER], rrt[EFFECT])
        self.assertGreater(rrt[CI95_UPPER], rrt[EFFECT])

    def test_no_treated_units_returns_nan(self):
        A0 = np.zeros_like(self.A)
        out = compute_tmle_rrt(A0, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertTrue(np.isnan(out[EFFECT]))
        self.assertTrue(np.isnan(out[EFFECT_treated]))
        self.assertTrue(np.isnan(out[EFFECT_untreated]))

    def test_no_control_units_returns_nan(self):
        """E[Y(0) | A=1] is unidentified without controls, so no finite RRT."""
        A1 = np.ones_like(self.A)
        out = compute_tmle_rrt(A1, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertTrue(np.isnan(out[EFFECT]))
        self.assertTrue(np.isnan(out[EFFECT_treated]))
        self.assertTrue(np.isnan(out[EFFECT_untreated]))

    def test_initial_effect_is_taken_over_the_treated(self):
        rrt = self._rrt()
        treated = self.A == 1
        m1 = float(self.Y1_hat[treated].mean())
        m0 = float(self.Y0_hat[treated].mean())
        self.assertAlmostEqual(rrt[INITIAL_EFFECT_treated], m1, places=12)
        self.assertAlmostEqual(rrt[INITIAL_EFFECT_untreated], m0, places=12)
        self.assertAlmostEqual(rrt[INITIAL_EFFECT], m1 / m0, places=12)
        self.assertAlmostEqual(
            rrt[ADJUSTMENT_untreated],
            rrt[EFFECT_untreated] - m0,
            places=12,
        )

    def test_clipping_changes_estimate_and_standard_error(self):
        plain = self._rrt()
        clipped = compute_tmle_rrt(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, clip_percentile=0.9
        )
        self.assertNotAlmostEqual(plain[EFFECT], clipped[EFFECT], places=6)
        self.assertNotAlmostEqual(plain[STD_ERR], clipped[STD_ERR], places=6)


if __name__ == "__main__":
    unittest.main()
