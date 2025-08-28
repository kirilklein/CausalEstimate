import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
    estimate_fluctuation_parameter,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
)
from CausalEstimate.estimators.functional.utils import (
    compute_clever_covariate_ate,
    compute_clever_covariate_att,
)
from CausalEstimate.utils.constants import EFFECT
from tests.helpers.setup import TestEffectBase


class TestTMLEFunctions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter(self):
        H = compute_clever_covariate_ate(self.A, self.ps)
        epsilon = estimate_fluctuation_parameter(H, self.Y, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATT_Functions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter_att(self):
        H = compute_clever_covariate_att(self.A, self.ps)
        epsilon = estimate_fluctuation_parameter(H, self.Y, self.Yhat)
        self.assertIsInstance(epsilon, float)
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATE_base(TestEffectBase):
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
