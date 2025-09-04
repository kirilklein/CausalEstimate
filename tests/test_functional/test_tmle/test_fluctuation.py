import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    estimate_fluctuation_parameter,
)

from CausalEstimate.estimators.functional.utils import (
    compute_clever_covariate_ate,
    compute_clever_covariate_att,
)
from tests.helpers.setup import TestEffectBase


class TestTMLEFunctions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter(self):
        H = compute_clever_covariate_ate(self.A, self.ps)
        epsilon = estimate_fluctuation_parameter(H, self.Y, self.Yhat)
        self.assertIsInstance(epsilon, float)
        # Check that epsilon is a finite number
        self.assertTrue(np.isfinite(epsilon))


class TestTMLE_ATT_Functions(TestEffectBase):
    """Basic tests for TMLE functions"""

    def test_estimate_fluctuation_parameter_att(self):
        H = compute_clever_covariate_att(self.A, self.ps)
        epsilon = estimate_fluctuation_parameter(H, self.Y, self.Yhat)
        self.assertIsInstance(epsilon, float)
        self.assertTrue(np.isfinite(epsilon))


if __name__ == "__main__":
    unittest.main()
