import unittest

from CausalEstimate.estimators.ipw import IPW
from CausalEstimate.utils.constants import EFFECT, OUTCOME_COL, PS_COL, TREATMENT_COL
from tests.helpers.setup import ContinuousEffectBase, TestEffectBase


class TestIPW(TestEffectBase):
    def test_compute_ipw_ate(self):
        ipw = IPW(
            effect_type="ATE",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
        )
        ate_ipw = ipw.compute_effect(self.data)
        self.assertAlmostEqual(ate_ipw[EFFECT], self.true_ate, delta=0.1)


class TestIPWContinuousOutcome(ContinuousEffectBase):
    def test_ate_matches_statsmodels(self):
        result = IPW(effect_type="ATE", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.sm_te.ipw().effect[0], places=5)
        self.assertAlmostEqual(result[EFFECT], self.true_ate, delta=0.1)

    def test_att_matches_statsmodels(self):
        result = IPW(effect_type="ATT", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(
            result[EFFECT], self.sm_te.ipw(effect_group=1).effect[0], places=5
        )
        self.assertAlmostEqual(result[EFFECT], self.true_att, delta=0.1)

    def test_rr_rejects_continuous_outcome(self):
        with self.assertRaises(ValueError):
            IPW(effect_type="RR", outcome_col=OUTCOME_COL).compute_effect(self.data)


if __name__ == "__main__":
    unittest.main()
