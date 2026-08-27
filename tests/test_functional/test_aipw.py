import unittest
import numpy as np
from CausalEstimate.estimators.functional.aipw import compute_aipw_ate, compute_aipw_att
from CausalEstimate.estimators.functional.ipw import compute_ipw_ate
from CausalEstimate.utils.constants import EFFECT
from tests.helpers.setup import TestEffectBase


class TestComputeAIPWATE(TestEffectBase):
    """Basic tests for AIPW estimators"""

    def test_invalid_input_shapes(self):
        # Test for mismatched input shapes
        A = np.array([1, 0, 1])
        Y = np.array([3, 1, 4])
        ps = np.array([0.8, 0.6])  # Mismatched length
        Y0_hat = np.array([2, 1.5, 3])
        Y1_hat = np.array([3.5, 2.0, 4.5])

        # Ensure that an exception is raised for mismatched input shapes
        with self.assertRaises(ValueError):
            compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat)


class TestAIPW_ATE_base(TestEffectBase):
    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.02)

    def test_constant_outcome_model_reduces_to_ipw(self):
        for c in (0.0, 0.3, 1.0):
            const = np.full_like(self.Y, c, dtype=float)
            ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, const, const)[EFFECT]
            ate_ipw = compute_ipw_ate(self.A, self.Y, self.ps)[EFFECT]
            self.assertAlmostEqual(ate_aipw, ate_ipw, places=12, msg=f"c={c}")

        


class TestAIPW_ATE_base_stabilized(TestEffectBase):
    def test_compute_aipw_ate_stabilized(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.03)


class TestAIPW_ATE_ps_misspecified(TestAIPW_ATE_base):
    alpha = [0.1, 0.2, -0.3, 10]

    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.03)


class TestAIPW_ATE_outcome_model_misspecified(TestAIPW_ATE_base):
    beta = [
        0.5,
        10,
        0.6,
        0.3,
        10,
    ]  # if the ps is correct, there is no adjustment, thus outcome model does not matter in this case.

    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.01)


class TestAIPW_ATE_outcome_and_ps_model_misspecified(TestAIPW_ATE_base):
    beta = [
        0.5,
        10,
        0.6,
        0.3,
        10,
    ]  # if the ps is correct, there is no adjustment, thus outcome model does not matter in this case.
    alpha = [0.1, 0.2, -0.3, 10]

    def test_compute_aipw_ate(self):
        ate_aipw = compute_aipw_ate(self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat)
        self.assertNotAlmostEqual(ate_aipw[EFFECT], self.true_ate, delta=0.05)


class TestAIPW_ATT_base(TestEffectBase):
    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.03)


class TestAIPW_ATT_outcome_model_misspecified(TestAIPW_ATT_base):
    beta = [
        0.5,
        0.8,
        -0.6,
        0.3,
        5,
    ]  # if the ps is correct, there is no adjustment, thus outcome model does not matter in this case.

    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.01)


class TestAIPW_ATT_ps_misspecified(TestAIPW_ATT_base):
    alpha = [
        0.1,
        0.2,
        -0.3,
        10,
        5,
    ]  # evem though the ps is misspecified, the adjustment gives as a correct effect

    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.01)


class TestAIPW_ATT_PS_misspecified_and_OutcomeModel_misspecified(TestAIPW_ATT_base):
    alpha = [0.1, 0.2, -0.3, 5]
    beta = [0.5, 0.8, -0.6, 0.3, 5]

    def test_compute_aipw_att(self):
        att_aipw = compute_aipw_att(self.A, self.Y, self.ps, self.Y0_hat)
        self.assertNotAlmostEqual(att_aipw[EFFECT], self.true_att, delta=0.01)


# Run the unittests
if __name__ == "__main__":
    unittest.main()
