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
from tests.helpers.setup import TestEffectBase


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


if __name__ == "__main__":
    unittest.main()
