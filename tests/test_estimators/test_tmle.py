import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_atc,
    compute_tmle_rrt,
)
from CausalEstimate.estimators.tmle import TMLE
from CausalEstimate.utils.constants import (
    OUTCOME_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
    PS_COL,
    TREATMENT_COL,
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
    STD_ERR,
    CI95_LOWER,
    CI95_UPPER,
)
from tests.helpers.setup import ContinuousEffectBase, TestEffectBase


class TestTMLE(TestEffectBase):
    def test_compute_tmle_ate(self):
        tmle = TMLE(
            effect_type="ATE",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
        )
        ate_tmle = tmle.compute_effect(self.data)
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.01)

    def test_compute_tmle_rr(self):
        """
        The Risk Ratio goes through the same per-arm targeting step as the ATE,
        so it must recover the truth just as tightly. A single fluctuation
        targeting the difference leaves each arm mean biased, and those biases
        do not cancel in a ratio.
        """
        tmle = TMLE(
            effect_type="RR",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
        )
        rr_tmle = tmle.compute_effect(self.data)

        self.assertAlmostEqual(rr_tmle[EFFECT], self.true_rr, delta=0.02)
        # The reported ratio is exactly the ratio of the two targeted arm means.
        self.assertAlmostEqual(
            rr_tmle[EFFECT], rr_tmle[EFFECT_treated] / rr_tmle[EFFECT_untreated]
        )
        # The CI is built on the log scale, so it stays positive and brackets
        # the estimate multiplicatively rather than symmetrically.
        self.assertGreater(rr_tmle[CI95_LOWER], 0)
        self.assertLess(rr_tmle[CI95_LOWER], rr_tmle[EFFECT])
        self.assertGreater(rr_tmle[CI95_UPPER], rr_tmle[EFFECT])


class TestTMLEContinuousOutcome(ContinuousEffectBase):
    def test_ate_recovers_truth(self):
        result = TMLE(effect_type="ATE", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.true_ate, delta=0.1)
        self.assertAlmostEqual(
            result[EFFECT], result[EFFECT_treated] - result[EFFECT_untreated]
        )
        self.assertAlmostEqual(
            result[CI95_UPPER] - result[CI95_LOWER], 2 * 1.96 * result[STD_ERR]
        )
        self.assertLess(result[CI95_LOWER], result[EFFECT])
        self.assertGreater(result[CI95_UPPER], result[EFFECT])

    def test_att_recovers_truth(self):
        result = TMLE(effect_type="ATT", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.true_att, delta=0.1)

    def test_y_bounds_wider_than_data(self):
        y = self.data[OUTCOME_COL]
        result = TMLE(
            effect_type="ATE",
            outcome_col=OUTCOME_COL,
            y_bounds=(y.min() - 5, y.max() + 5),
        )
        result = result.compute_effect(self.data)
        self.assertAlmostEqual(result[EFFECT], self.true_ate, delta=0.1)

    def test_invalid_y_bounds_raise(self):
        for bounds in [(0, 1), (0, np.inf), (5, 5)]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                TMLE(
                    effect_type="ATE", outcome_col=OUTCOME_COL, y_bounds=bounds
                ).compute_effect(self.data)

    def test_atc_recovers_truth(self):
        result = TMLE(effect_type="ATC", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.true_atc, delta=0.1)

    def test_rr_rejects_continuous_outcome(self):
        with self.assertRaises(ValueError):
            TMLE(effect_type="RR", outcome_col=OUTCOME_COL).compute_effect(self.data)


class TestTMLERRTEffectType(TestEffectBase):
    """
    The class dispatches "RRT" to compute_tmle_rrt. The estimator itself is
    covered in tests/test_functional/test_tmle/test_rrt.py.
    """

    def _tmle(self):
        return TMLE(
            effect_type="RRT",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
        )

    def test_rrt_dispatches_to_the_functional_estimator(self):
        got = self._tmle().compute_effect(self.data)
        want = compute_tmle_rrt(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertEqual(got.keys(), want.keys())
        for key in want:
            self.assertAlmostEqual(got[key], want[key], places=12, msg=key)

    def test_rrt_rejects_continuous_outcome(self):
        df = self.data.copy()
        df[OUTCOME_COL] = df[OUTCOME_COL].astype(float) + 0.5
        with self.assertRaises(ValueError):
            self._tmle().compute_effect(df)


class TestTMLEATCEffectType(TestEffectBase):
    """
    The class dispatches "ATC" to compute_tmle_atc, covered in
    tests/test_functional/test_atc.py. Every key is compared: on this fixture
    a Y0_hat / Y1_hat mix-up leaves the targeted means unchanged and shows
    only in the initial effect.
    """

    def test_atc_dispatches_to_the_functional_estimator(self):
        got = TMLE(
            effect_type="ATC",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
        ).compute_effect(self.data)
        want = compute_tmle_atc(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertEqual(got.keys(), want.keys())
        for key in want:
            self.assertAlmostEqual(got[key], want[key], places=12, msg=key)


if __name__ == "__main__":
    unittest.main()
