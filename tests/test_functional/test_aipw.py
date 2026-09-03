import unittest

import numpy as np
import pandas as pd

from CausalEstimate.estimators.aipw import AIPW
from CausalEstimate.estimators.functional.aipw import compute_aipw_ate, compute_aipw_att
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_weights,
)
from CausalEstimate.utils.constants import EFFECT, EFFECT_treated, EFFECT_untreated
from tests.helpers.setup import TestEffectBase


class TestComputeAIPWATE(TestEffectBase):
    """Basic tests for AIPW estimators"""

    def test_invalid_input_shapes(self):
        # Test for mismatched input shapes
        A = np.array([1, 0, 1])
        Y = np.array([3, 1, 4])
        ps = np.array([0.8, 0.6])  # Mismatched length
        Y0_hat = np.array([2, 1.5, 3])
        Y1_hat = np.array([3.5, 2.0, 4.5])

        # Ensure that an exception is raised for mismatched input shapes
        with self.assertRaises(ValueError):
            compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat)


class TestAIPW_ATE_base(TestEffectBase):
    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.02)

    def test_constant_outcome_model_reduces_to_ipw(self):
        for c in (0.0, 0.3, 1.0):
            const = np.full_like(self.Y, c, dtype=float)
            ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, const, const)[EFFECT]
            ate_ipw = compute_ipw_ate(self.A, self.Y, self.ps)[EFFECT]
            self.assertAlmostEqual(ate_aipw, ate_ipw, places=12, msg=f"c={c}")

    def test_matches_ipw_plus_augmentation(self):
        A, Y, ps, Y0_hat, Y1_hat = self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat
        W = compute_ipw_weights(A, ps, weight_type="ATE")
        w1, w0 = A * W, (1 - A) * W
        augmentation = (
            (w1 / w1.mean() - 1) * Y1_hat - (w0 / w0.mean() - 1) * Y0_hat
        ).mean()
        expected = compute_ipw_ate(A, Y, ps)[EFFECT] - augmentation
        actual = compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat)[EFFECT]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_empty_group_warns_and_returns_nan(self):
        A = np.ones_like(self.A)
        with self.assertWarns(RuntimeWarning):
            ate = compute_aipw_ate(A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertTrue(np.isnan(ate[EFFECT]))


class TestAIPW_ATE_base_stabilized(TestEffectBase):
    def test_compute_aipw_ate_stabilized(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.03)


class TestAIPW_ATE_ps_misspecified(TestAIPW_ATE_base):
    alpha = [0.1, 0.2, -0.3, 10]

    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.03)


class TestAIPW_ATE_outcome_model_misspecified(TestAIPW_ATE_base):
    beta = [
        0.5,
        10,
        0.6,
        0.3,
        10,
    ]  # if the ps is correct, there is no adjustment, thus outcome model does not matter in this case.

    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.01)


class TestAIPW_ATE_outcome_and_ps_model_misspecified(TestAIPW_ATE_base):
    beta = [
        0.5,
        10,
        0.6,
        0.3,
        10,
    ]  # if the ps is correct, there is no adjustment, thus outcome model does not matter in this case.
    alpha = [0.1, 0.2, -0.3, 10]

    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertNotAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.05)


class TestAIPW_ATT_base(TestEffectBase):
    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.03)


class TestAIPW_ATT_outcome_model_misspecified(TestAIPW_ATT_base):
    beta = [
        0.5,
        0.8,
        -0.6,
        0.3,
        5,
    ]  # if the ps is correct, there is no adjustment, thus outcome model does not matter in this case.

    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.01)


class TestAIPW_ATT_ps_misspecified(TestAIPW_ATT_base):
    alpha = [
        0.1,
        0.2,
        -0.3,
        10,
        5,
    ]  # evem though the ps is misspecified, the adjustment gives as a correct effect

    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.01)


class TestAIPW_ATT_PS_misspecified_and_OutcomeModel_misspecified(TestAIPW_ATT_base):
    alpha = [0.1, 0.2, -0.3, 5]
    beta = [0.5, 0.8, -0.6, 0.3, 5]

    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertNotAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.01)


# Run the unittests
if __name__ == "__main__":
    unittest.main()


class TestAIPWClippingAndPotentialOutcomes(unittest.TestCase):
    """Issue #92: AIPW must honour clip_percentile/eps and report mu_1 / mu_0."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 4000
        # U-shaped propensities, so a real fraction sit near 0 and 1 and
        # clipping has something to bite on
        self.ps = rng.beta(0.4, 0.4, n).clip(1e-4, 1 - 1e-4)
        self.A = rng.binomial(1, self.ps)
        self.Y1_hat = rng.uniform(0.3, 0.7, n)
        self.Y0_hat = rng.uniform(0.1, 0.5, n)
        self.Y = rng.binomial(
            1, np.where(self.A == 1, self.Y1_hat, self.Y0_hat)
        ).astype(float)

    def test_default_clipping_is_a_no_op(self):
        """clip_percentile=1 must reproduce the unclipped estimate exactly."""
        a = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        b = compute_aipw_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, clip_percentile=1
        )
        self.assertAlmostEqual(a[EFFECT], b[EFFECT], places=12)

    def test_clipping_changes_the_estimate(self):
        """With extreme propensity scores, clipping must actually do something."""
        unclipped = compute_aipw_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, clip_percentile=1
        )[EFFECT]
        clipped = compute_aipw_ate(
            self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat, clip_percentile=0.9
        )[EFFECT]
        self.assertNotAlmostEqual(unclipped, clipped, places=6)

    def test_ate_reports_potential_outcome_means(self):
        r = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        for key in (EFFECT_treated, EFFECT_untreated):
            self.assertIn(key, r)
        self.assertAlmostEqual(
            r[EFFECT], r[EFFECT_treated] - r[EFFECT_untreated], places=12
        )

    def test_att_reports_potential_outcome_means(self):
        r = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        for key in (EFFECT_treated, EFFECT_untreated):
            self.assertIn(key, r)
        # mu_1 for the ATT is just the observed treated mean
        self.assertAlmostEqual(r[EFFECT_treated], self.Y[self.A == 1].mean(), places=12)
        self.assertAlmostEqual(
            r[EFFECT], r[EFFECT_treated] - r[EFFECT_untreated], places=12
        )

    def test_att_matches_the_previous_weight_construction(self):
        """The switch to compute_ipw_weights(weight_type="ATT") must not move
        the estimate. Reproduces the old ad-hoc weights inline."""
        w = self.ps / (1 - self.ps)
        control_factor = (1 - self.A) * w
        S = self.A / (self.A == 1).sum() - control_factor / control_factor.sum()
        old = (S * (self.Y - self.Y0_hat)).sum()
        new = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)[EFFECT]
        self.assertAlmostEqual(old, new, places=8)

    def test_estimator_class_threads_the_parameters(self):
        df = pd.DataFrame(
            {
                "treatment": self.A,
                "outcome": self.Y,
                "ps": self.ps,
                "probas_t1": self.Y1_hat,
                "probas_t0": self.Y0_hat,
            }
        )
        plain = AIPW(effect_type="ATE").compute_effect(df)
        clipped = AIPW(effect_type="ATE", clip_percentile=0.9).compute_effect(df)
        self.assertNotAlmostEqual(plain[EFFECT], clipped[EFFECT], places=6)
