import unittest

from CausalEstimate.estimators.survival.ipcw import IPCW
from tests.helpers.setup_survival import TestSurvivalBase


class TestIPCW(TestSurvivalBase):
    def test_ipcw(self):
        ipcw = IPCW()
        ipcw.compute_effect(self.data, self.ps_col, self.survival_prob_col, self.time_col)
        


if __name__ == "__main__":
    unittest.main()
