import unittest

import numpy as np
from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    estimate_fluctuation_parameter,
    update_ate_estimate,
)
from tests.test_functional.base import TestEffectBase


class TestTMLEFunctions(TestEffectBase):
    def test_estimate_fluctuation_parameter(self):
        epsilon = estimate_fluctuation_parameter(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))

    def test_update_ate_estimate(self):
        epsilon = 0.1  # Arbitrary small fluctuation parameter
        ate = update_ate_estimate(self.ps, self.Y0_hat, self.Y1_hat, epsilon)
        self.assertIsInstance(ate, float)
        # Check that ate is within a reasonable range
        self.assertTrue(-5 <= ate <= 5)

    def test_compute_tmle_ate(self):
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertIsInstance(ate_tmle, float)
        # The true ATE is 2; check if the estimate is close
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.1)

    def test_compute_tmle_ate_edge_cases(self):
        # Test with ps very close to 0 or 1
        ps_edge = self.ps.copy()
        ps_edge[ps_edge < 0.01] = 0.01
        ps_edge[ps_edge > 0.99] = 0.99
        ate_tmle = compute_tmle_ate(
            self.A, self.Y, ps_edge, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertIsInstance(ate_tmle, float)
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.15)


if __name__ == "__main__":
    unittest.main()
