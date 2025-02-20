import unittest

from CausalEstimate.estimators.ipw import IPW
from tests.helpers.setup import TestEffectBase
from CausalEstimate.utils.constants import OUTCOME_COL, PS_COL, TREATMENT_COL


class TestIPW(TestEffectBase):
    def test_compute_ipw_ate(self):
        ipw = IPW(
            effect_type="ATE",
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
        )
        ate_ipw = ipw.compute_effect(self.data)
        self.assertAlmostEqual(ate_ipw, self.true_ate, delta=0.1)


if __name__ == "__main__":
    unittest.main()
