import unittest

import numpy as np

from CausalEstimate.estimators.tmle import TMLE
from CausalEstimate.utils.constants import (
    OUTCOME_COL,
    PROBAS_COL,
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
            probas_col=PROBAS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
        )
        ate_tmle = tmle.compute_effect(self.data)
        self.assertAlmostEqual(ate_tmle[EFFECT], self.true_ate, delta=0.01)


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

    def test_rr_rejects_continuous_outcome(self):
        with self.assertRaises(ValueError):
            TMLE(effect_type="RR", outcome_col=OUTCOME_COL).compute_effect(self.data)


if __name__ == "__main__":
    unittest.main()
