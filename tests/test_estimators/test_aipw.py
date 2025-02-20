import unittest

from CausalEstimate.estimators.aipw import AIPW
from tests.helpers.setup import TestEffectBase


class TestAIPW(TestEffectBase):
    def test_compute_aipw_ate(self):
        aipw = AIPW(
            effect_type="ATE",
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="ps",
            probas_t1_col="probas_t1",
            probas_t0_col="probas_t0",
        )
        ate_aipw = aipw.compute_effect(self.data)
        self.assertAlmostEqual(ate_aipw, self.true_ate, delta=0.01)


if __name__ == "__main__":
    unittest.main()
