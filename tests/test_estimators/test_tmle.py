import unittest

from CausalEstimate.estimators.tmle import TMLE
from tests.helpers.setup import TestEffectBase


class TestTMLE(TestEffectBase):
    def test_compute_tmle_ate(self):
        tmle = TMLE(effect_type="ATE")
        ate_tmle = tmle.compute_effect(
            self.data, "treatment", "outcome", "ps", "Yhat", "Y1_hat", "Y0_hat"
        )
        self.assertAlmostEqual(ate_tmle, self.true_ate, delta=0.01)


if __name__ == "__main__":
    unittest.main()
