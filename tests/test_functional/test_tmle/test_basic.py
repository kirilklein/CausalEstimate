import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
)
from CausalEstimate.utils.constants import (
    ADJUSTMENT_untreated,
    EFFECT,
    EFFECT_untreated,
    INITIAL_EFFECT,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
)
from tests.helpers.setup import TestEffectBase


class TestTMLE_ATE_base(TestEffectBase):
    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_ATE_base2(TestEffectBase):
    alpha = [1, -0.2, -0.3]
    beta = [0.1, 0.4, 0.6, -2]

    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_ATE_base3(TestEffectBase):
    alpha = [-1, 2, -0.3]
    beta = [-1, 0.4, 0.6, -2]

    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_RR(TestEffectBase):
    def test_compute_tmle_rr(self):
        rr_tmle = compute_tmle_rr(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(rr_tmle[EFFECT], self.true_rr, delta=1)


class TestTMLE_ATT(TestEffectBase):
    def test_compute_tmle_att(self):
        att_tmle = compute_tmle_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(att_tmle[EFFECT], self.true_att, delta=0.02)

    def test_initial_effect_is_taken_over_the_treated(self):
        """The initial effect must refer to the same population as the ATT."""
        att = compute_tmle_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        treated = self.A == 1
        m1 = float(self.Y1_hat[treated].mean())
        m0 = float(self.Y0_hat[treated].mean())
        self.assertAlmostEqual(att[INITIAL_EFFECT_treated], m1, places=12)
        self.assertAlmostEqual(att[INITIAL_EFFECT_untreated], m0, places=12)
        self.assertAlmostEqual(att[INITIAL_EFFECT], m1 - m0, places=12)
        self.assertAlmostEqual(
            att[ADJUSTMENT_untreated], att[EFFECT_untreated] - m0, places=12
        )

    def test_empty_arm_returns_nan(self):
        """Without controls the ATT is unidentified, not the untargeted fit."""
        for A in (np.ones_like(self.A), np.zeros_like(self.A)):
            with self.subTest(all_treated=bool(A[0])):
                att = compute_tmle_att(A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
                self.assertTrue(np.isnan(att[EFFECT]))


class TestTMLE_ATT_bounded(TestEffectBase):
    def test_att_is_bounded(self):
        att_tmle = compute_tmle_att(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertLessEqual(att_tmle[EFFECT], 1)
        self.assertGreaterEqual(att_tmle[EFFECT], -1)


if __name__ == "__main__":
    unittest.main()
