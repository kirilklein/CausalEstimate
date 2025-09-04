import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
)
from CausalEstimate.utils.constants import EFFECT
from tests.helpers.setup import TestEffectBase


class TestTMLE_ATE_stabilized(TestEffectBase):
    """Checks if the stabilized TMLE ATE can recover the true effect."""

    def test_compute_tmle_ate_stabilized(self):
        ate_tmle = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.02)


class TestTMLE_ATT_stabilized(TestEffectBase):
    """
    Checks if the stabilized TMLE ATT can recover the true effect.
    NOTE: This assumes a `stabilized` flag has been added to `compute_tmle_att`
    in the same way as `compute_tmle_ate`.
    """

    def test_compute_tmle_att_stabilized(self):
        att_tmle = compute_tmle_att(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )
        self.assertAlmostEqual(att_tmle[EFFECT], self.true_att, delta=0.02)


class TestTMLEStabilizedVsUnstabilized(TestEffectBase):
    """Comprehensive comparison of stabilized vs unstabilized TMLE"""

    def test_stabilized_unstabilized_ate_comparison(self):
        """Compare stabilized vs unstabilized ATE estimates"""
        ate_unstabilized = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=False,
        )
        ate_stabilized = compute_tmle_ate(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )

        # Both should be finite
        self.assertTrue(np.isfinite(ate_unstabilized[EFFECT]))
        self.assertTrue(np.isfinite(ate_stabilized[EFFECT]))

        # Effects should be reasonably close (within 20% for well-behaved data)
        relative_diff = abs(ate_stabilized[EFFECT] - ate_unstabilized[EFFECT]) / abs(
            ate_unstabilized[EFFECT]
        )
        self.assertLess(
            relative_diff,
            0.2,
            f"Stabilized and unstabilized estimates differ too much: {relative_diff}",
        )

    def test_stabilized_unstabilized_att_comparison(self):
        """Compare stabilized vs unstabilized ATT estimates"""
        att_unstabilized = compute_tmle_att(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=False,
        )
        att_stabilized = compute_tmle_att(
            self.A,
            self.Y,
            self.ps,
            self.Y0_hat,
            self.Y1_hat,
            self.Yhat,
            stabilized=True,
        )

        # Both should be finite
        self.assertTrue(np.isfinite(att_unstabilized[EFFECT]))
        self.assertTrue(np.isfinite(att_stabilized[EFFECT]))

        # Effects should be reasonably close
        relative_diff = abs(att_stabilized[EFFECT] - att_unstabilized[EFFECT]) / abs(
            att_unstabilized[EFFECT]
        )
        self.assertLess(
            relative_diff,
            0.2,
            f"Stabilized and unstabilized ATT estimates differ too much: {relative_diff}",
        )


if __name__ == "__main__":
    unittest.main()
