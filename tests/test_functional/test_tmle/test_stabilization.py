import unittest
from typing import List

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


class TestTMLEStabilizationBenefit(TestEffectBase):
    """
    Demonstrates that stabilization reduces variance for the TMLE estimator
    by bootstrapping from a single, high-variance data simulation.
    """

    # Override alpha from TestEffectBase to create a high-variance scenario
    alpha: List[float] = [0.5, -2.5, 3.0, 0]
    # Use a larger sample for a more stable bootstrap base
    n: int = 5000

    def _get_bootstrap_standard_error(
        self, n_replicates: int, stabilized: bool
    ) -> float:
        """
        Calculates the standard error of the TMLE ATE estimate via bootstrap.
        It relies on the full data created by TestEffectBase.setUpClass.
        """
        rng = np.random.default_rng(self.seed)
        n_obs = len(self.A)
        bootstrap_ates = []

        for _ in range(n_replicates):
            # Create a bootstrap sample by drawing indices with replacement
            indices = rng.choice(n_obs, size=n_obs, replace=True)

            # Resample all necessary arrays for TMLE
            A_boot = self.A[indices]
            Y_boot = self.Y[indices]
            ps_boot = self.ps[indices]
            Y0_hat_boot = self.Y0_hat[indices]
            Y1_hat_boot = self.Y1_hat[indices]
            Yhat_boot = self.Yhat[indices]

            # Calculate the ATE on the resampled data using the actual TMLE function
            ate_boot = compute_tmle_ate(
                A_boot,
                Y_boot,
                ps_boot,
                Y0_hat_boot,
                Y1_hat_boot,
                Yhat_boot,
                stabilized=stabilized,
            )

            if not np.isnan(ate_boot[EFFECT]):
                bootstrap_ates.append(ate_boot[EFFECT])

        # The standard deviation of the bootstrap estimates is our standard error
        return np.std(bootstrap_ates)

    def test_stabilization_reduces_bootstrap_variance(self):
        """
        Asserts that the bootstrap standard error is smaller for the stabilized TMLE estimator.
        """
        n_replicates = 100  # Keep lower for speed, increase for precision

        # --- Estimate Standard Error for both estimators ---
        se_unstabilized = self._get_bootstrap_standard_error(
            n_replicates=n_replicates, stabilized=False
        )
        se_stabilized = self._get_bootstrap_standard_error(
            n_replicates=n_replicates, stabilized=True
        )

        print(
            f"\n[TMLE Stabilization Benefit] Unstabilized ATE SE: {se_unstabilized:.4f}"
        )
        print(f"[TMLE Stabilization Benefit] Stabilized ATE SE:   {se_stabilized:.4f}")

        # --- The Definitive Assertion ---
        self.assertLess(
            se_stabilized,
            se_unstabilized,
            "Stabilized TMLE should yield a lower bootstrap standard error, indicating reduced variance.",
        )


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
