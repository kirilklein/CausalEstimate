"""
Tests for the influence-curve variance layer.

The analytic standard errors rest on two properties, and the tests here check
both directly rather than only spot-checking numbers:

1. Each influence curve is mean-zero at the reported point estimate. This is
   what makes the sample variance of the curve an estimate of the sampling
   variance, and it is the property that breaks first if a weight, a sign or a
   normaliser is wrong.
2. The resulting standard error agrees with a nonparametric bootstrap of the
   same estimator, for every estimator/estimand pair.
"""

import unittest

import numpy as np

from CausalEstimate.estimators.functional.aipw import (
    compute_aipw_ate,
    compute_aipw_att,
)
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_att,
    compute_ipw_risk_ratio,
    compute_ipw_risk_ratio_treated,
)
from CausalEstimate.estimators.functional.tmle import compute_tmle_ate, compute_tmle_rr
from CausalEstimate.estimators.functional.tmle_att import compute_tmle_att
from CausalEstimate.estimators.functional.utils import (
    compute_ipw_weights,
    target_outcome_models,
)
from CausalEstimate.estimators.functional.variance import (
    _compute_ic_log_ratio,
    _compute_ic_mu,
    _summarise_ic,
    compute_ci,
    compute_ci_aipw,
    compute_ci_ipw,
)
from CausalEstimate.utils.constants import (
    CI95_LOWER,
    CI95_UPPER,
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
    STD_ERR,
)
from tests.helpers.setup import generate_simulation_data

ALPHA = [-0.2, 0.5, -0.5]
BETA = [0.1, 0.4, 0.6, -2]

# Every estimator/estimand pair that reports an analytic standard error, as
# (label, callable taking the simulation dict, is_ratio).
ESTIMATORS = [
    ("ipw/ATE", lambda d: compute_ipw_ate(d["A"], d["Y"], d["ps"]), False),
    ("ipw/ATT", lambda d: compute_ipw_att(d["A"], d["Y"], d["ps"]), False),
    ("ipw/RR", lambda d: compute_ipw_risk_ratio(d["A"], d["Y"], d["ps"]), True),
    (
        "ipw/RRT",
        lambda d: compute_ipw_risk_ratio_treated(d["A"], d["Y"], d["ps"]),
        True,
    ),
    (
        "aipw/ATE",
        lambda d: compute_aipw_ate(d["A"], d["Y"], d["ps"], d["Y0_hat"], d["Y1_hat"]),
        False,
    ),
    (
        "aipw/ATT",
        lambda d: compute_aipw_att(d["A"], d["Y"], d["ps"], d["Y0_hat"]),
        False,
    ),
    (
        "tmle/ATE",
        lambda d: compute_tmle_ate(d["A"], d["Y"], d["ps"], d["Y0_hat"], d["Y1_hat"]),
        False,
    ),
    (
        "tmle/ATT",
        lambda d: compute_tmle_att(d["A"], d["Y"], d["ps"], d["Y0_hat"], d["Y1_hat"]),
        False,
    ),
    (
        "tmle/RR",
        lambda d: compute_tmle_rr(d["A"], d["Y"], d["ps"], d["Y0_hat"], d["Y1_hat"]),
        True,
    ),
]


def _sim(n=2000, seed=0, noise_level=0.0):
    return generate_simulation_data(
        n=n, alpha=ALPHA, beta=BETA, noise_level=noise_level, seed=seed
    )


class TestInfluenceCurveHelpers(unittest.TestCase):
    """Unit tests for the pieces the three public entry points share."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        n = 500
        self.A = self.rng.binomial(1, 0.5, n)
        self.Y = self.rng.binomial(1, 0.4, n).astype(float)
        self.Q = self.rng.uniform(0.2, 0.8, n)
        self.w = np.where(self.A == 1, self.rng.uniform(1.0, 3.0, n), 0.0)

    def test_normalised_curve_is_mean_zero_at_the_hajek_mean(self):
        """The Hajek weighted mean is exactly the centre that zeroes the curve."""
        mu = (self.w * self.Y).sum() / self.w.sum()
        ic = _compute_ic_mu(self.Y, self.w, np.full(self.Y.shape, mu), mu)
        self.assertAlmostEqual(float(ic.mean()), 0.0, places=12)

    def test_constant_q_recovers_the_plain_hajek_curve(self):
        """With Q == mu the plug-in term drops out, leaving w (Y - mu)/mean(w)."""
        mu = (self.w * self.Y).sum() / self.w.sum()
        ic = _compute_ic_mu(self.Y, self.w, np.full(self.Y.shape, mu), mu)
        expected = self.w * (self.Y - mu) / self.w.mean()
        np.testing.assert_allclose(ic, expected, rtol=1e-12)

    def test_normalise_flag_only_rescales_the_residual_term(self):
        mu = float(self.Q.mean())
        norm = _compute_ic_mu(self.Y, self.w, self.Q, mu, normalize=True)
        raw = _compute_ic_mu(self.Y, self.w, self.Q, mu, normalize=False)
        residual = self.w * (self.Y - self.Q)
        np.testing.assert_allclose(
            norm - raw, residual / self.w.mean() - residual, rtol=1e-12
        )

    def test_treated_restricted_curve_centres_on_a_over_p(self):
        """
        A/P(A=1) weighting is what makes an ATT arm curve mean-zero, but
        only at the augmented mean the AIPW estimator actually reports: the
        treated mean of Q plus the weighted residual.
        """
        A_over_p = self.A / self.A.mean()
        w = np.where(self.A == 0, self.w.max() * 0.5, 0.0)  # a control-arm weight
        mu = float((A_over_p * self.Q).mean()) + (w * (self.Y - self.Q)).sum() / w.sum()
        ic = _compute_ic_mu(self.Y, w, self.Q, mu, A_over_p=A_over_p)
        self.assertAlmostEqual(float(ic.mean()), 0.0, places=12)

    def test_zero_weight_arm_gives_nan_rather_than_dividing_by_zero(self):
        ic = _compute_ic_mu(self.Y, np.zeros_like(self.w), self.Q, 0.5)
        self.assertTrue(np.all(np.isnan(ic)))

    def test_nan_mean_propagates(self):
        ic = _compute_ic_mu(self.Y, self.w, self.Q, np.nan)
        self.assertTrue(np.all(np.isnan(ic)))

    def test_log_ratio_is_the_delta_method_combination(self):
        ic1 = self.rng.normal(size=50)
        ic0 = self.rng.normal(size=50)
        got = _compute_ic_log_ratio(ic1, ic0, 0.4, 0.2)
        np.testing.assert_allclose(got, ic1 / 0.4 - ic0 / 0.2, rtol=1e-12)

    def test_log_ratio_rejects_non_positive_arm_means(self):
        ic = np.ones(10)
        for mu_1, mu_0 in [(0.0, 0.3), (0.3, 0.0), (-0.2, 0.3), (np.nan, 0.3)]:
            with self.subTest(mu_1=mu_1, mu_0=mu_0):
                out = _compute_ic_log_ratio(ic, ic, mu_1, mu_0)
                self.assertTrue(np.all(np.isnan(out)))


class TestSummariseIC(unittest.TestCase):
    """The CI must be reconstructible from the reported STD_ERR."""

    def setUp(self):
        self.ic = np.random.default_rng(1).normal(size=400)

    def test_difference_ci_is_symmetric_around_psi(self):
        out = _summarise_ic("ATE", 0.25, self.ic)
        self.assertAlmostEqual(out[CI95_LOWER], 0.25 - 1.96 * out[STD_ERR], places=12)
        self.assertAlmostEqual(out[CI95_UPPER], 0.25 + 1.96 * out[STD_ERR], places=12)

    def test_ratio_ci_is_the_exponentiated_log_scale_interval(self):
        """STD_ERR is on the log scale for ratios; the CI is its exponentiation."""
        for effect_type in ("RR", "RRT"):
            with self.subTest(effect_type=effect_type):
                out = _summarise_ic(effect_type, 1.4, self.ic)
                se = out[STD_ERR]
                self.assertAlmostEqual(
                    out[CI95_LOWER], np.exp(np.log(1.4) - 1.96 * se), places=12
                )
                self.assertAlmostEqual(
                    out[CI95_UPPER], np.exp(np.log(1.4) + 1.96 * se), places=12
                )
                self.assertGreater(out[CI95_LOWER], 0.0)

    def test_nan_inputs_give_nan_results(self):
        for psi, ic in [(np.nan, self.ic), (0.3, np.full(10, np.nan))]:
            with self.subTest(psi=psi):
                out = _summarise_ic("ATE", psi, ic)
                self.assertTrue(all(np.isnan(v) for v in out.values()))


class TestComputeCIValidation(unittest.TestCase):
    """Argument handling on the three public entry points."""

    def setUp(self):
        self.d = _sim(n=400, seed=3)

    def test_rr_requires_the_arm_weights(self):
        with self.assertRaises(ValueError):
            compute_ci(
                effect_type="RR",
                psi=1.2,
                Q_star_1=self.d["Y1_hat"],
                Q_star_0=self.d["Y0_hat"],
                Y=self.d["Y"],
                A=self.d["A"],
                Yhat_star=self.d["Yhat"],
            )

    def test_unsupported_effect_type_raises(self):
        W = compute_ipw_weights(self.d["A"], self.d["ps"])
        with self.assertRaises(ValueError):
            compute_ci_ipw("NOPE", 1.0, self.d["Y"], self.d["A"], W, 0.5, 0.4)
        with self.assertRaises(ValueError):
            compute_ci_aipw(
                "NOPE",
                1.0,
                self.d["Y"],
                self.d["A"],
                W,
                self.d["Y1_hat"],
                self.d["Y0_hat"],
                0.5,
                0.4,
            )

    def test_empty_input_returns_nan(self):
        empty = np.array([])
        out = compute_ci_ipw("ATE", 0.1, empty, empty, empty, 0.5, 0.4)
        self.assertTrue(all(np.isnan(v) for v in out.values()))
        out = compute_ci_aipw("ATE", 0.1, empty, empty, empty, empty, empty, 0.5, 0.4)
        self.assertTrue(all(np.isnan(v) for v in out.values()))

    def test_no_treated_units_returns_nan_for_the_att(self):
        n = len(self.d["Y"])
        A0 = np.zeros(n, dtype=int)
        W = compute_ipw_weights(A0, self.d["ps"], weight_type="ATT")
        out = compute_ci_aipw(
            "ATT", 1.0, self.d["Y"], A0, W, None, self.d["Y0_hat"], 0.5, 0.4
        )
        self.assertTrue(all(np.isnan(v) for v in out.values()))


class TestReportedCIIsConsistentWithStdErr(unittest.TestCase):
    """Every estimator must report a CI that its own STD_ERR reproduces."""

    @classmethod
    def setUpClass(cls):
        cls.d = _sim(n=3000, seed=11)

    def test_all_estimators(self):
        for label, fn, is_ratio in ESTIMATORS:
            with self.subTest(label=label):
                r = fn(self.d)
                se, psi = r[STD_ERR], r[EFFECT]
                self.assertTrue(np.isfinite(se) and se > 0, f"{label}: bad SE")
                if is_ratio:
                    lo = np.exp(np.log(psi) - 1.96 * se)
                    hi = np.exp(np.log(psi) + 1.96 * se)
                else:
                    lo, hi = psi - 1.96 * se, psi + 1.96 * se
                self.assertAlmostEqual(r[CI95_LOWER], lo, places=10)
                self.assertAlmostEqual(r[CI95_UPPER], hi, places=10)

    def test_ratio_estimators_report_a_consistent_ratio_of_arm_means(self):
        for label, fn, is_ratio in ESTIMATORS:
            if not is_ratio:
                continue
            with self.subTest(label=label):
                r = fn(self.d)
                self.assertAlmostEqual(
                    r[EFFECT], r[EFFECT_treated] / r[EFFECT_untreated], places=10
                )


class TestAnalyticSEMatchesBootstrap(unittest.TestCase):
    """
    The influence curve is a linear approximation to the estimator's sampling
    distribution, so its standard error should agree with a nonparametric
    bootstrap of the same estimator. This is the test that would catch a wrong
    weight, sign or normaliser; the tolerance is loose because the bootstrap
    itself is noisy at this replicate count.
    """

    n = 1500
    n_boot = 120
    seed = 5

    @classmethod
    def setUpClass(cls):
        cls.d = _sim(n=cls.n, seed=cls.seed)

    def _bootstrap_se(self, fn, is_ratio):
        rng = np.random.default_rng(99)
        n = len(self.d["Y"])
        keys = ("A", "Y", "ps", "Y0_hat", "Y1_hat", "Yhat")
        estimates = []
        for _ in range(self.n_boot):
            idx = rng.integers(0, n, n)
            resampled = {k: np.asarray(self.d[k])[idx] for k in keys}
            psi = fn(resampled)[EFFECT]
            if np.isfinite(psi) and (not is_ratio or psi > 0):
                estimates.append(np.log(psi) if is_ratio else psi)
        return float(np.std(estimates, ddof=1))

    def test_all_estimators(self):
        for label, fn, is_ratio in ESTIMATORS:
            with self.subTest(label=label):
                analytic = fn(self.d)[STD_ERR]
                boot = self._bootstrap_se(fn, is_ratio)
                ratio = analytic / boot
                self.assertTrue(
                    0.75 <= ratio <= 1.35,
                    f"{label}: analytic SE {analytic:.5f} vs bootstrap "
                    f"{boot:.5f} (ratio {ratio:.3f})",
                )


class TestAgainstExplicitEIF(unittest.TestCase):
    """
    Restate each efficient influence function from the estimating equations and
    check the reported standard error reproduces it exactly.

    A bootstrap comparison is too coarse to pin down signs and normalisers: a
    negated arm weight, for instance, leaves the curve mean-zero and shifts the
    SE by well under a percent when the outcome model is correct. Writing the
    EIF out independently is what fixes those choices.
    """

    @classmethod
    def setUpClass(cls):
        cls.d = _sim(n=3000, seed=23)

    @staticmethod
    def _se(ic):
        return float(np.sqrt(np.var(ic, ddof=1) / len(ic)))

    def _check(self, label, reported, ic):
        self.assertAlmostEqual(
            float(ic.mean()), 0.0, places=8, msg=f"{label}: not centred"
        )
        self.assertAlmostEqual(reported, self._se(ic), places=10, msg=label)

    def test_ipw_curves(self):
        d = self.d
        A, Y, ps = d["A"], d["Y"], d["ps"]
        for label, wt, fn in [
            ("ipw/ATE", "ATE", compute_ipw_ate),
            ("ipw/ATT", "ATT", compute_ipw_att),
            ("ipw/RR", "ATE", compute_ipw_risk_ratio),
            ("ipw/RRT", "ATT", compute_ipw_risk_ratio_treated),
        ]:
            with self.subTest(label=label):
                r = fn(A, Y, ps)
                W = compute_ipw_weights(A, ps, weight_type=wt)
                w1, w0 = A * W, (1 - A) * W
                mu_1, mu_0 = r[EFFECT_treated], r[EFFECT_untreated]
                # Hajek curve per arm: w (Y - mu) / mean(w)
                ic_1 = w1 * (Y - mu_1) / w1.mean()
                ic_0 = w0 * (Y - mu_0) / w0.mean()
                ic = ic_1 / mu_1 - ic_0 / mu_0 if "RR" in label else ic_1 - ic_0
                self._check(label, r[STD_ERR], ic)

    def test_aipw_curves(self):
        d = self.d
        A, Y, ps, Q1, Q0 = d["A"], d["Y"], d["ps"], d["Y1_hat"], d["Y0_hat"]
        p = A.mean()

        with self.subTest(label="aipw/ATE"):
            r = compute_aipw_ate(A, Y, ps, Q0, Q1)
            W = compute_ipw_weights(A, ps, weight_type="ATE")
            w1, w0 = A * W, (1 - A) * W
            mu_1, mu_0 = r[EFFECT_treated], r[EFFECT_untreated]
            ic_1 = w1 * (Y - Q1) / w1.mean() + (Q1 - mu_1)
            ic_0 = w0 * (Y - Q0) / w0.mean() + (Q0 - mu_0)
            self._check("aipw/ATE", r[STD_ERR], ic_1 - ic_0)

        with self.subTest(label="aipw/ATT"):
            r = compute_aipw_att(A, Y, ps, Q0)
            W = compute_ipw_weights(A, ps, weight_type="ATT")
            w0 = (1 - A) * W
            mu_1, mu_0 = r[EFFECT_treated], r[EFFECT_untreated]
            # mu_1 is the raw treated mean; mu_0 carries the augmentation
            # and centres on A/p because it is a mean over the treated.
            ic_1 = (A / p) * (Y - mu_1)
            ic_0 = w0 * (Y - Q0) / w0.mean() + (A / p) * (Q0 - mu_0)
            self._check("aipw/ATT", r[STD_ERR], ic_1 - ic_0)

    def test_tmle_rr_curve(self):
        """
        The targeting step solves each arm's score equation, so the residual
        term carries no Hajek normaliser -- and each arm's own weight enters
        with a POSITIVE sign. Only the difference covariate H = w1 - w0 is
        signed.
        """
        d = self.d
        A, Y, ps, Q1, Q0 = d["A"], d["Y"], d["ps"], d["Y1_hat"], d["Y0_hat"]

        r = compute_tmle_rr(A, Y, ps, Q0, Q1)
        res = target_outcome_models(A, Y, ps, Q1, Q0, effect_type="RR")
        mu_1, mu_0 = r[EFFECT_treated], r[EFFECT_untreated]
        ic_1 = res.w1 * (Y - res.Q_star_1) + (res.Q_star_1 - mu_1)
        ic_0 = res.w0 * (Y - res.Q_star_0) + (res.Q_star_0 - mu_0)
        self._check("tmle/RR", r[STD_ERR], ic_1 / mu_1 - ic_0 / mu_0)

    def test_a_negated_arm_weight_is_detected(self):
        """
        Guards the regression this class exists for: the control-arm weight
        must not be negated. Mean-zero survives the flip, so only comparing
        against the explicit curve catches it.
        """
        d = self.d
        A, Y, ps, Q1, Q0 = d["A"], d["Y"], d["ps"], d["Y1_hat"], d["Y0_hat"]
        r = compute_tmle_rr(A, Y, ps, Q0, Q1)
        res = target_outcome_models(A, Y, ps, Q1, Q0, effect_type="RR")
        mu_1, mu_0 = r[EFFECT_treated], r[EFFECT_untreated]
        ic_1 = res.w1 * (Y - res.Q_star_1) + (res.Q_star_1 - mu_1)
        flipped = -res.w0 * (Y - res.Q_star_0) + (res.Q_star_0 - mu_0)
        wrong_se = self._se(ic_1 / mu_1 - flipped / mu_0)
        # Still mean-zero, which is why the weaker checks miss it...
        self.assertAlmostEqual(float(flipped.mean()), 0.0, places=8)
        # ...but not the standard error the estimator actually reports.
        self.assertNotAlmostEqual(r[STD_ERR], wrong_se, places=10)


class TestClippingKeepsEstimateAndSEConsistent(unittest.TestCase):
    """
    The SE must be computed from the same (clipped) weights as the estimate.
    Clipping changes both; a variance layer that recomputed its own weights
    would leave the SE unchanged.
    """

    @classmethod
    def setUpClass(cls):
        cls.d = _sim(n=3000, seed=13)

    def test_clipping_moves_the_standard_error(self):
        cases = [
            ("ipw/ATE", lambda c: compute_ipw_ate(*self._args(), clip_percentile=c)),
            (
                "ipw/RRT",
                lambda c: compute_ipw_risk_ratio_treated(
                    *self._args(), clip_percentile=c
                ),
            ),
            (
                "aipw/ATE",
                lambda c: compute_aipw_ate(
                    *self._args(), self.d["Y0_hat"], self.d["Y1_hat"], clip_percentile=c
                ),
            ),
        ]
        for label, fn in cases:
            with self.subTest(label=label):
                base = fn(1.0)[STD_ERR]
                clipped = fn(0.9)[STD_ERR]
                self.assertTrue(np.isfinite(clipped))
                self.assertNotAlmostEqual(
                    base, clipped, places=9, msg=f"{label}: SE ignored clipping"
                )

    def _args(self):
        return self.d["A"], self.d["Y"], self.d["ps"]


if __name__ == "__main__":
    unittest.main()
