import unittest
import warnings
from unittest import mock

import numpy as np
from scipy.special import expit, logit
from statsmodels.genmod.generalized_linear_model import GLM

from CausalEstimate.estimators.functional.utils import (
    compute_arm_weights,
    compute_initial_effect,
    compute_ipw_weights,
    check_score_equations,
    estimate_arm_fluctuation,
    target_outcome_models,
    _one_step_fluctuation
)
from CausalEstimate.utils.constants import (
    INITIAL_EFFECT,
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
    INITIAL_EFFECT_treated,
)


class TestTMLEUtils(unittest.TestCase):
    """Unit tests for the TMLE utility functions."""

    def setUp(self):
        """Set up a simple, fixed dataset for consistent testing."""
        self.A = np.array([1, 1, 0, 0])
        self.Y = np.array([1, 0, 1, 1])
        self.ps = np.array([0.8, 0.6, 0.4, 0.2])
        self.Yhat = np.array([0.7, 0.6, 0.5, 0.4])
        self.Y1_hat = np.array([0.75, 0.65, 0.55, 0.45])
        self.Y0_hat = np.array([0.65, 0.55, 0.45, 0.35])
        self.Q_star_1 = self.Y1_hat + 0.05
        self.Q_star_0 = self.Y0_hat - 0.05

    def test_compute_initial_effect(self):
        """Test calculation of initial effect and adjustments."""
        results = compute_initial_effect(
            self.Y1_hat, self.Y0_hat, self.Q_star_1, self.Q_star_0, rr=False
        )

        expected_initial_effect = self.Y1_hat.mean() - self.Y0_hat.mean()
        expected_adj_1 = 0.05
        expected_adj_0 = -0.05

        self.assertAlmostEqual(results[INITIAL_EFFECT], expected_initial_effect)
        self.assertAlmostEqual(results[ADJUSTMENT_treated], expected_adj_1)
        self.assertAlmostEqual(results[ADJUSTMENT_untreated], expected_adj_0)
        self.assertIn(INITIAL_EFFECT_treated, results)

    def test_compute_initial_effect_rr_zero_denominator(self):
        """Test risk ratio calculation when the denominator is zero."""
        Y0_hat_zero = np.array([0.0, 0.0, 0.0])
        Y1_hat_one = np.array([1.0, 1.0, 1.0])

        with self.assertWarns(RuntimeWarning):
            results = compute_initial_effect(
                Y1_hat_one, Y0_hat_zero, Y1_hat_one, Y0_hat_zero, rr=True
            )
            self.assertEqual(results[INITIAL_EFFECT], np.inf)


class TestArmWeights(unittest.TestCase):
    """
    Tests for `compute_arm_weights`, the per-arm replacement for the single
    signed clever covariate.
    """

    def setUp(self):
        self.A = np.array([1, 1, 0, 0])
        self.ps = np.array([0.8, 0.6, 0.4, 0.2])

    def test_ate_weights_are_per_arm_and_nonnegative(self):
        """Each arm's weights are its own IPW weights and zero off-arm."""
        w1, w0 = compute_arm_weights(self.A, self.ps, "ATE", clip_percentile=1.0)

        np.testing.assert_allclose(w1, [1 / 0.8, 1 / 0.6, 0.0, 0.0], rtol=1e-6)
        np.testing.assert_allclose(w0, [0.0, 0.0, 1 / 0.6, 1 / 0.8], rtol=1e-6)

        # Both weights are non-negative: the direction of each fluctuation comes
        # from the root of its score equation, not from a sign on the covariate.
        self.assertTrue(np.all(w1 >= 0))
        self.assertTrue(np.all(w0 >= 0))

    def test_rr_weights_match_ate_weights(self):
        """RR shares the ATE targeting weights; only the combination differs."""
        w1_ate, w0_ate = compute_arm_weights(self.A, self.ps, "ATE")
        w1_rr, w0_rr = compute_arm_weights(self.A, self.ps, "RR")
        np.testing.assert_array_equal(w1_ate, w1_rr)
        np.testing.assert_array_equal(w0_ate, w0_rr)

    def test_att_weights_match_the_treated_subpopulation_formula(self):
        """ATT reweights each arm to the treated subpopulation."""
        p_treated = 0.5
        w1, w0 = compute_arm_weights(self.A, self.ps, "ATT", clip_percentile=1.0)

        np.testing.assert_allclose(w1, [1 / p_treated, 1 / p_treated, 0.0, 0.0])
        np.testing.assert_allclose(
            w0,
            [
                0.0,
                0.0,
                self.ps[2] / (p_treated * (1 - self.ps[2])),
                self.ps[3] / (p_treated * (1 - self.ps[3])),
            ],
            rtol=1e-6,
        )

    def test_h_property_reproduces_the_signed_clever_covariate(self):
        """
        w1 - w0 is the signed covariate the ATE/ATT variance step expects, and
        TargetingResult.H exposes exactly the weights that were fitted.
        """
        A, ps = self.A, self.ps
        p_treated = float(np.mean(A == 1))
        expected = {
            "ATE": A / ps - (1 - A) / (1 - ps),
            "ATT": A / p_treated - (1 - A) * ps / (p_treated * (1 - ps)),
        }
        for effect_type, expected_H in expected.items():
            with self.subTest(effect_type=effect_type):
                w1, w0 = compute_arm_weights(A, ps, effect_type)
                result = target_outcome_models(
                    A,
                    np.array([1.0, 0.0, 1.0, 1.0]),
                    ps,
                    np.full(4, 0.6),
                    np.full(4, 0.4),
                    effect_type=effect_type,
                )
                np.testing.assert_allclose(result.H, w1 - w0)
                np.testing.assert_allclose(result.H, expected_H, rtol=1e-6)

    def test_invalid_arguments_raise(self):
        """An unknown estimand or an out-of-range clip_percentile is rejected."""
        with self.assertRaises(ValueError):
            compute_arm_weights(self.A, self.ps, "NOT_AN_EFFECT")
        for clip_percentile in (0.0, -0.1, 1.5):
            with self.subTest(clip_percentile=clip_percentile):
                with self.assertRaises(ValueError):
                    compute_arm_weights(self.A, self.ps, "ATE", clip_percentile)

    def test_clipping_is_applied_within_each_arm(self):
        """
        Clipping uses a per-arm quantile, so an extreme control weight cannot
        drag down the treated arm's threshold (or vice versa).
        """
        A = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        ps = np.array([0.5, 0.5, 0.5, 0.5, 0.99, 0.5, 0.5, 0.5])

        w1_unclipped, w0_unclipped = compute_arm_weights(A, ps, "ATE", 1.0)
        w1_clipped, w0_clipped = compute_arm_weights(A, ps, "ATE", 0.75)

        # The treated arm's weights are all identical, so its own quantile
        # leaves them untouched.
        np.testing.assert_array_equal(w1_unclipped, w1_clipped)
        # The control arm has one extreme weight, which is clipped down.
        self.assertLess(w0_clipped.max(), w0_unclipped.max())
        self.assertTrue(np.all(np.isfinite(w0_clipped)))

    def test_extreme_weights_warn(self):
        """
        Propensity scores near 0 or 1 make the fluctuation unstable, so the
        weights warn rather than failing silently.
        """
        ps_extreme = np.array([0.9999, 0.0001])
        A_extreme = np.array([0, 1])
        with self.assertWarnsRegex(RuntimeWarning, "targeting weights above"):
            compute_arm_weights(A_extreme, ps_extreme, "ATE", clip_percentile=1.0)

    def test_every_valid_clip_percentile_gives_finite_weights(self):
        """Any clip_percentile the validator accepts must produce usable weights."""
        for effect_type in ("ATE", "ATT"):
            for clip_percentile in (0.1, 0.5, 0.9, 1.0):
                with self.subTest(
                    effect_type=effect_type, clip_percentile=clip_percentile
                ):
                    w1, w0 = compute_arm_weights(
                        self.A, self.ps, effect_type, clip_percentile
                    )
                    self.assertTrue(np.all(np.isfinite(w1)))
                    self.assertTrue(np.all(np.isfinite(w0)))

    def test_no_treated_units_gives_zero_weights_with_a_warning(self):
        """The ATT is undefined without treated units; targeting is a no-op."""
        A = np.zeros(4, dtype=int)
        with self.assertWarnsRegex(RuntimeWarning, "No treated subjects"):
            w1, w0 = compute_arm_weights(A, self.ps, "ATT")
        np.testing.assert_array_equal(w1, np.zeros(4))
        np.testing.assert_array_equal(w0, np.zeros(4))


class TestEstimateArmFluctuation(unittest.TestCase):
    """
    Tests for `estimate_arm_fluctuation`, the per-arm weighted logistic
    fluctuation that replaced the single combined clever-covariate GLM fit.
    """

    def setUp(self):
        self.A = np.array([1, 1, 1, 0, 0, 0])
        self.Y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        self.ps = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.45])
        self.Q = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
        self.w = self.A / self.ps

    def test_root_solves_the_score_equation(self):
        """The returned epsilon zeroes sum(w * (Y - expit(logit(Q) + eps)))."""
        epsilon = estimate_arm_fluctuation(self.Y, self.Q, self.w, "treated")
        Q_star = expit(logit(self.Q) + epsilon)
        score = float(np.sum(self.w * (self.Y - Q_star)))
        self.assertAlmostEqual(score, 0.0, places=10)

    def test_score_is_decreasing_so_the_root_is_unique(self):
        """A shift either side of the root moves the score in one direction only."""
        epsilon = estimate_arm_fluctuation(self.Y, self.Q, self.w, "treated")

        def score(shift):
            return float(
                np.sum(self.w * (self.Y - expit(logit(self.Q) + epsilon + shift)))
            )

        self.assertGreater(score(-0.5), 0.0)
        self.assertLess(score(0.5), 0.0)

    def test_separated_arm_above_clamps_to_plus_max_shift(self):
        """Every weighted outcome equal to 1 pushes Q* to 1, i.e. epsilon to +inf."""
        Y_all_events = np.where(self.A == 1, 1.0, 0.0)
        with self.assertWarnsRegex(RuntimeWarning, "unbounded above"):
            epsilon = estimate_arm_fluctuation(
                Y_all_events, self.Q, self.w, "treated", max_shift=30.0
            )
        self.assertEqual(epsilon, 30.0)

    def test_separated_arm_below_clamps_to_minus_max_shift(self):
        """Symmetrically, no weighted events pushes epsilon to -inf."""
        Y_no_events = np.where(self.A == 1, 0.0, 1.0)
        with self.assertWarnsRegex(RuntimeWarning, "unbounded below"):
            epsilon = estimate_arm_fluctuation(
                Y_no_events, self.Q, self.w, "treated", max_shift=30.0
            )
        self.assertEqual(epsilon, -30.0)

    def test_arm_with_no_weight_is_skipped(self):
        """An empty arm returns epsilon = 0 rather than fitting an empty GLM."""
        with self.assertWarnsRegex(RuntimeWarning, "No units carry weight"):
            epsilon = estimate_arm_fluctuation(self.Y, self.Q, np.zeros(6), "treated")
        self.assertEqual(epsilon, 0.0)

    def test_boundary_predictions_stay_finite(self):
        """Q at 0 or 1 is bounded by q_eps before the logit, so no inf leaks out."""
        Q_boundary = np.array([0.0, 1.0, 0.5, 0.0, 1.0, 0.5])
        epsilon = estimate_arm_fluctuation(self.Y, Q_boundary, self.w, "treated")
        self.assertTrue(np.isfinite(epsilon))

    def test_large_weights_keep_the_exact_glm_fit(self):
        """
        A well-behaved arm carrying large IPW weights must not fall back.

        statsmodels compares the absolute deviance change to `tol`, and
        freq_weights scale the deviance, so this fit reports
        `converged=False` at tol=1e-12 even though epsilon is exact. Judging
        the fit by `fit.converged` would discard it and return the cruder
        one-step on exactly the low-propensity data TMLE is for.
        """
        Y = np.array([1.0, 0.0, 1.0, 0.0])
        Q = np.full(4, 0.5)
        w = np.array([1e3, 1e6, 1e6, 1e6])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            epsilon = estimate_arm_fluctuation(Y, Q, w, "treated")

        score = float(np.sum(w * (Y - expit(logit(Q) + epsilon))))
        self.assertLess(abs(score), 1e-8 * np.sum(w))

        one_step = _one_step_fluctuation(Y, logit(Q), w, "treated", 30.0)
        self.assertNotAlmostEqual(epsilon, one_step, places=3)

    def test_failed_glm_falls_back_to_the_one_step_estimate(self):
        """A GLM that cannot be fitted hands over to the Newton one-step."""
        with mock.patch.object(GLM, "fit", side_effect=np.linalg.LinAlgError):
            with self.assertWarnsRegex(RuntimeWarning, "one-step estimate"):
                epsilon = estimate_arm_fluctuation(self.Y, self.Q, self.w, "treated")

        score = float(np.sum(self.w * (self.Y - self.Q)))
        information = float(np.sum(self.w * self.Q * (1 - self.Q)))
        self.assertAlmostEqual(epsilon, score / information, places=12)

    def test_one_step_estimate_moves_the_score_toward_zero(self):
        """The fallback is a real step, not just a finite number."""
        with mock.patch.object(GLM, "fit", side_effect=np.linalg.LinAlgError):
            with self.assertWarns(RuntimeWarning):
                epsilon = estimate_arm_fluctuation(self.Y, self.Q, self.w, "treated")

        def score(shift):
            return abs(
                float(np.sum(self.w * (self.Y - expit(logit(self.Q) + shift))))
            )

        self.assertLess(score(epsilon), score(0.0))

    def test_one_step_estimate_is_clamped_to_max_shift(self):
        """Near-zero information cannot send the step off to an overflowing shift."""
        Q_boundary = np.full(6, 1e-12)
        with mock.patch.object(GLM, "fit", side_effect=np.linalg.LinAlgError):
            with self.assertWarns(RuntimeWarning):
                epsilon = estimate_arm_fluctuation(
                    self.Y, Q_boundary, self.w, "treated", max_shift=30.0
                )
        self.assertLessEqual(abs(epsilon), 30.0)

    def test_one_step_estimate_without_information_returns_zero(self):
        """
        Zero information has no step to take, so the arm is left untargeted.

        Tested on the helper directly: the separation guard in
        `estimate_arm_fluctuation` clamps before the GLM is ever reached on a
        degenerate arm, so this is a defensive branch rather than a reachable
        one.
        """
        with self.assertWarnsRegex(RuntimeWarning, "zero information"):
            epsilon = _one_step_fluctuation(
                self.Y, np.zeros(6), np.zeros(6), "treated", 30.0
            )
        self.assertEqual(epsilon, 0.0)


class TestTargetOutcomeModels(unittest.TestCase):
    """
    Tests for `target_outcome_models`, the single entry point that fluctuates
    each arm separately.
    """

    def setUp(self):
        rng = np.random.default_rng(20240917)
        n = 400
        self.A = rng.binomial(1, 0.5, n)
        self.ps = np.clip(rng.uniform(0.2, 0.8, n), 0.01, 0.99)
        self.Y = rng.binomial(1, 0.3 + 0.3 * self.A, n).astype(float)
        self.Y1_hat = np.clip(rng.uniform(0.3, 0.7, n), 0.01, 0.99)
        self.Y0_hat = np.clip(rng.uniform(0.2, 0.6, n), 0.01, 0.99)

    def _target(self, effect_type):
        return target_outcome_models(
            self.A, self.Y, self.ps, self.Y1_hat, self.Y0_hat, effect_type=effect_type
        )

    def test_both_arm_score_equations_are_solved(self):
        """
        The point of the per-arm fluctuation: each arm mean is individually
        targeted, so both score equations hold, not just their difference.
        """
        for effect_type in ("ATE", "RR", "ATT"):
            with self.subTest(effect_type=effect_type):
                result = self._target(effect_type)
                score_1, score_0 = check_score_equations(result, self.Y)
                self.assertAlmostEqual(score_1, 0.0, places=10)
                self.assertAlmostEqual(score_0, 0.0, places=10)

    def test_targeted_predictions_match_the_reported_epsilons(self):
        """Q* is exactly the initial fit shifted by the epsilon that is returned."""
        result = self._target("ATE")
        np.testing.assert_allclose(
            result.Q_star_1, expit(logit(self.Y1_hat) + result.epsilon_1), rtol=1e-10
        )
        np.testing.assert_allclose(
            result.Q_star_0, expit(logit(self.Y0_hat) + result.epsilon_0), rtol=1e-10
        )

    def test_yhat_star_is_the_arm_specific_prediction(self):
        """
        Yhat_star is derived from Q_star_1/Q_star_0 rather than fluctuated on
        its own, so it cannot drift out of step with them.
        """
        result = self._target("ATE")
        np.testing.assert_array_equal(
            result.Yhat_star, np.where(self.A == 1, result.Q_star_1, result.Q_star_0)
        )

    def test_constant_nuisances_recover_the_unadjusted_arm_means(self):
        """
        With a constant propensity score and a constant outcome model, the
        targeted arm means must equal the observed arm means exactly. A single
        fluctuation targeting only the difference leaves each arm mean biased,
        so this pins down per-arm targeting.
        """
        ps_constant = np.full_like(self.ps, 0.5)
        Y1_constant = np.full_like(self.Y1_hat, 0.4)
        Y0_constant = np.full_like(self.Y0_hat, 0.4)

        result = target_outcome_models(
            self.A, self.Y, ps_constant, Y1_constant, Y0_constant, effect_type="ATE"
        )

        self.assertAlmostEqual(
            float(result.Q_star_1.mean()), float(self.Y[self.A == 1].mean()), places=10
        )
        self.assertAlmostEqual(
            float(result.Q_star_0.mean()), float(self.Y[self.A == 0].mean()), places=10
        )

    def test_check_score_equations_warns_on_an_untargeted_result(self):
        """The diagnostic warns instead of asserting, so -O cannot silence it."""
        result = self._target("ATE")
        untargeted = result._replace(Q_star_1=self.Y1_hat, Q_star_0=self.Y0_hat)
        with self.assertWarnsRegex(RuntimeWarning, "is not solved"):
            check_score_equations(untargeted, self.Y)


class TestIPWWeightFunction(unittest.TestCase):
    """
    Directly tests the `compute_ipw_weights` function to ensure logic is correct.
    """

    @classmethod
    def setUpClass(cls):
        cls.A = np.array([1, 1, 0, 0])
        cls.ps = np.array([0.8, 0.4, 0.5, 0.2])
        cls.pi = 0.5

    def test_att_weights(self):
        weights = compute_ipw_weights(self.A, self.ps, weight_type="ATT")
        stabilization_factor = (1 - self.pi) / self.pi
        expected = np.array(
            [
                1.0,
                1.0,
                (0.5 / 0.5) * stabilization_factor,
                (0.2 / 0.8) * stabilization_factor,
            ]
        )
        np.testing.assert_allclose(weights, expected)


if __name__ == "__main__":
    unittest.main()
