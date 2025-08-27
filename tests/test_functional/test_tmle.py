import unittest
from typing import List

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
    estimate_fluctuation_parameter,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
    estimate_fluctuation_parameter_att,
)
from CausalEstimate.utils.constants import EFFECT
from tests.helpers.setup import TestEffectBase


class TestTMLEFunctions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter(self):
        epsilon = estimate_fluctuation_parameter(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATT_Functions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter_att(self):
        epsilon = estimate_fluctuation_parameter_att(self.A, self.Y, self.ps, self.Yhat)
        self.assertIsInstance(epsilon, float)
        self.assertTrue(np.isfinite(epsilon))


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


class TestTMLE_ATT_bounded(TestEffectBase):
    def test_att_is_bounded(self):
        att_tmle = compute_tmle_att(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, self.Yhat
        )
        self.assertLessEqual(att_tmle[EFFECT], 1)
        self.assertGreaterEqual(att_tmle[EFFECT], -1)


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


if __name__ == "__main__":
    unittest.main()
