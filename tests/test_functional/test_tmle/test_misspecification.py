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


if __name__ == "__main__":
    unittest.main()
