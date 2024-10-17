import unittest

from CausalEstimate.estimators.ipw import IPW
from tests.helpers.setup import TestEffectBase


class TestIPW(TestEffectBase):
    def test_compute_ipw_ate(self):
        ipw = IPW(effect_type="ATE")
        ate_ipw = ipw.compute_effect(self.data, "treatment", "outcome", "ps")
        self.assertAlmostEqual(ate_ipw, self.true_ate, delta=0.1)


if __name__ == "__main__":
    unittest.main()
