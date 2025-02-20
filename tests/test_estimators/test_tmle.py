import unittest

from CausalEstimate.estimators.tmle import TMLE
from tests.helpers.setup import TestEffectBase


class TestTMLE(TestEffectBase):
    def test_compute_tmle_ate(self):
        tmle = TMLE(
            effect_type="ATE",
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="ps",
            probas_col="probas",
            probas_t1_col="probas_t1",
            probas_t0_col="probas_t0",
        )
        ate_tmle = tmle.compute_effect(self.data)
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.01)


if __name__ == "__main__":
    unittest.main()
