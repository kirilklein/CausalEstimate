"""
Tests for the IPW, AIPW and TMLE estimators of the ATC.

Each ATC estimator is implemented as the ATT with the roles of the two arms
swapped, so the identity

    ATC(A, Y, ps, Y0_hat, Y1_hat) = -ATT(1 - A, Y, 1 - ps, Y1_hat, Y0_hat)

holds by construction. Checking it still pins down the wiring -- which arm's
predictions go where -- which the truth-recovery tests cannot: with a correct
propensity model, double robustness recovers the ATC even from the wrong
outcome predictions. The key mapping itself is unit-tested in
tests/test_functional/test_utils.py (TestATTResultAsATC), and the estimator
classes' ATC dispatch in tests/test_estimators.
"""

import unittest

import numpy as np

from CausalEstimate.estimators.functional.aipw import (
    compute_aipw_atc,
    compute_aipw_att,
)
from CausalEstimate.estimators.functional.ipw import compute_ipw_atc, compute_ipw_att
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_atc,
    compute_tmle_att,
)
from CausalEstimate.estimators.functional.utils import att_result_as_atc
from CausalEstimate.simulation.binary_simulation import (
    compute_expected_outcome,
    simulate_binary_data,
)
from CausalEstimate.utils.constants import (
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
    TREATMENT_COL,
)
from tests.helpers.setup import TestEffectBase


class _ATCFixture(TestEffectBase):
    """Simulated data, the true ATC, and every ATC estimator run on it."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        data = simulate_binary_data(
            cls.n, alpha=cls.alpha, beta=cls.beta, seed=cls.seed
        )
        controls = data[data[TREATMENT_COL] == 0]
        cls.true_atc = compute_expected_outcome(
            controls, cls.beta, 1
        ) - compute_expected_outcome(controls, cls.beta, 0)

    def _estimators(self, clip_percentile=1):
        """(name, ATC result, arm-swapped ATT result) for each estimator."""
        A, Y, ps, Y0, Y1 = self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat
        kw = {"clip_percentile": clip_percentile}
        return [
            (
                "IPW",
                compute_ipw_atc(A, Y, ps, **kw),
                compute_ipw_att(1 - A, Y, 1 - ps, **kw),
            ),
            (
                "AIPW",
                compute_aipw_atc(A, Y, ps, Y1, **kw),
                compute_aipw_att(1 - A, Y, 1 - ps, Y1, **kw),
            ),
            (
                "TMLE",
                compute_tmle_atc(A, Y, ps, Y0, Y1, **kw),
                compute_tmle_att(1 - A, Y, 1 - ps, Y1, Y0, **kw),
            ),
        ]

    def check_recovers_the_true_atc(self, delta):
        for name, atc, _ in self._estimators():
            with self.subTest(estimator=name):
                self.assertAlmostEqual(atc[EFFECT], self.true_atc, delta=delta)


class TestATC(_ATCFixture):
    def test_recovers_the_true_atc(self):
        # Tighter than the 0.010 gap between the true ATC and ATT here, so an
        # estimator that returned the ATT would fail.
        self.check_recovers_the_true_atc(delta=0.005)

    def test_is_the_att_with_the_arms_swapped(self):
        for clip in (1, 0.95):
            for name, atc, att in self._estimators(clip):
                with self.subTest(estimator=name, clip_percentile=clip):
                    self.assertAlmostEqual(atc[EFFECT], -att[EFFECT], places=12)
                    self.assertAlmostEqual(
                        atc[EFFECT_treated], att[EFFECT_untreated], places=12
                    )
                    self.assertAlmostEqual(
                        atc[EFFECT_untreated], att[EFFECT_treated], places=12
                    )
                    # Every key, not just the three above: on this fixture the
                    # outcome predictions differ by a constant logit shift,
                    # which TMLE's per-arm fluctuation absorbs, so a Y0_hat /
                    # Y1_hat mix-up shows only in the initial effect.
                    want = att_result_as_atc(att)
                    self.assertEqual(atc.keys(), want.keys())
                    for key in want:
                        self.assertAlmostEqual(atc[key], want[key], places=12)

    def test_empty_arm_returns_nan(self):
        estimators = {
            "IPW": lambda A: compute_ipw_atc(A, self.Y, self.ps),
            "AIPW": lambda A: compute_aipw_atc(A, self.Y, self.ps, self.Y1_hat),
            "TMLE": lambda A: compute_tmle_atc(
                A, self.Y, self.ps, self.Y0_hat, self.Y1_hat
            ),
        }
        for A in (np.ones_like(self.A), np.zeros_like(self.A)):
            for name, estimate in estimators.items():
                with self.subTest(estimator=name, all_treated=bool(A[0])):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out = estimate(A)
                    self.assertTrue(np.isnan(out[EFFECT]))


class TestATCOutcomeModelMisspecified(_ATCFixture):
    """A correct propensity model should rescue a wrong outcome model."""

    beta = [0.5, 0.8, -0.6, 0.3, 5]

    def test_recovers_the_true_atc(self):
        self.check_recovers_the_true_atc(delta=0.02)


if __name__ == "__main__":
    unittest.main()
