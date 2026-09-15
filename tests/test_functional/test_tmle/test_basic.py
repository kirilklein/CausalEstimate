import unittest

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
)
from CausalEstimate.utils.constants import EFFECT
from tests.helpers.setup import TestEffectBase


class TestTMLE_ATE_base(TestEffectBase):
    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_ATE_base2(TestEffectBase):
    alpha = [1, -0.2, -0.3]
    beta = [0.1, 0.4, 0.6, -2]

    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_ATE_base3(TestEffectBase):
    alpha = [-1, 2, -0.3]
    beta = [-1, 0.4, 0.6, -2]

    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_RR(TestEffectBase):
    def test_compute_tmle_rr(self):
        rr_tmle = compute_tmle_rr(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(rr_tmle[EFFECT], self.true_rr, delta=1)


class TestTMLE_ATT(TestEffectBase):
    def test_compute_tmle_att(self):
        att_tmle = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(att_tmle[EFFECT], self.true_att, delta=0.02)


class TestTMLE_ATT_bounded(TestEffectBase):
    def test_att_is_bounded(self):
        att_tmle = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertLessEqual(att_tmle[EFFECT], 1)
        self.assertGreaterEqual(att_tmle[EFFECT], -1)


if __name__ == "__main__":
    unittest.main()
