import unittest

import numpy as np
from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    estimate_fluctuation_parameter,
)
from tests.helpers.setup import TestEffectBase


class TestTMLEFunctions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter(self):
        epsilon = estimate_fluctuation_parameter(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATE_base(TestEffectBase):
    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.02)


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
        self.assertNotAlmostEqual(ate_tmle, self.true_ate, delta=0.1)


if __name__ == "__main__":
    unittest.main()
