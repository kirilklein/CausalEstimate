import unittest

from CausalEstimate.estimators.aipw import AIPW
from tests.helpers.setup import TestEffectBase


class TestAIPW(TestEffectBase):
    def test_compute_aipw_ate(self):
        aipw = AIPW(effect_type="ATE")
        ate_aipw = aipw.compute_effect(
            self.data, "treatment", "outcome", "ps", "Y1_hat", "Y0_hat"
        )
        self.assertAlmostEqual(ate_aipw, self.true_ate, delta=0.01)


if __name__ == "__main__":
    unittest.main()
