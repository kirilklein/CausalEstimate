import unittest

from CausalEstimate.estimators.aipw import AIPW
from CausalEstimate.utils.constants import (
    EFFECT,
    OUTCOME_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
    PS_COL,
    TREATMENT_COL,
)
from tests.helpers.setup import ContinuousEffectBase, TestEffectBase


class TestAIPW(TestEffectBase):
    def test_compute_aipw_ate(self):
        aipw = AIPW(
            effect_type="ATE",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
        )
        ate_aipw = aipw.compute_effect(self.data)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.01)


class TestAIPWContinuousOutcome(ContinuousEffectBase):
    def test_ate_matches_statsmodels(self):
        result = AIPW(effect_type="ATE", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.sm_te.aipw().effect[0], places=5)
        self.assertAlmostEqual(result[EFFECT], self.true_ate, delta=0.1)

    def test_att_recovers_truth(self):
        result = AIPW(effect_type="ATT", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.true_att, delta=0.1)

    def test_rr_rejects_continuous_outcome(self):
        with self.assertRaises(ValueError):
            AIPW(effect_type="RR", outcome_col=OUTCOME_COL).compute_effect(self.data)


if __name__ == "__main__":
    unittest.main()
