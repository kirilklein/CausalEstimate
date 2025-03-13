import unittest

from CausalEstimate.estimators.tmle import TMLE
from CausalEstimate.utils.constants import (
    OUTCOME_COL,
    PROBAS_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
    PS_COL,
    TREATMENT_COL,
    EFFECT,
)
from tests.helpers.setup import TestEffectBase


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


if __name__ == "__main__":
    unittest.main()
