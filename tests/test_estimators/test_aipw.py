import unittest

from CausalEstimate.estimators.aipw import AIPW
from CausalEstimate.estimators.functional.aipw import (
    compute_aipw_atc,
    compute_aipw_rr,
    compute_aipw_rrt,
)
from CausalEstimate.utils.constants import (
    EFFECT,
    OUTCOME_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
    PS_COL,
    TREATMENT_COL,
)
from tests.helpers.setup import ContinuousEffectBase, TestEffectBase


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


class TestAIPWContinuousOutcome(ContinuousEffectBase):
    def test_ate_matches_statsmodels(self):
        result = AIPW(effect_type="ATE", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.sm_te.aipw().effect[0], places=5)
        self.assertAlmostEqual(result[EFFECT], self.true_ate, delta=0.1)

    def test_att_recovers_truth(self):
        result = AIPW(effect_type="ATT", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.true_att, delta=0.1)

    def test_atc_recovers_truth(self):
        result = AIPW(effect_type="ATC", outcome_col=OUTCOME_COL).compute_effect(
            self.data
        )
        self.assertAlmostEqual(result[EFFECT], self.true_atc, delta=0.1)

    def test_rr_rejects_continuous_outcome(self):
        with self.assertRaises(ValueError):
            AIPW(effect_type="RR", outcome_col=OUTCOME_COL).compute_effect(self.data)


class TestAIPWEffectTypeDispatch(TestEffectBase):
    """
    The class dispatches "ATC", "RR" and "RRT" to their functional estimators,
    which are covered in tests/test_functional/test_atc.py and test_aipw.py.
    Exact equality also pins which outcome predictions are passed, which
    truth recovery cannot: double robustness hides a Y0_hat / Y1_hat mix-up.
    """

    def test_effect_types_dispatch_to_the_functional_estimators(self):
        A, Y, ps, Y0, Y1 = self.A, self.Y, self.ps, self.Y0_hat, self.Y1_hat
        cases = {
            "ATC": compute_aipw_atc(A, Y, ps, Y1),
            "RR": compute_aipw_rr(A, Y, ps, Y0, Y1),
            "RRT": compute_aipw_rrt(A, Y, ps, Y0),
        }
        for effect_type, want in cases.items():
            with self.subTest(effect_type=effect_type):
                got = AIPW(
                    effect_type=effect_type,
                    treatment_col=TREATMENT_COL,
                    outcome_col=OUTCOME_COL,
                    ps_col=PS_COL,
                    probas_t1_col=PROBAS_T1_COL,
                    probas_t0_col=PROBAS_T0_COL,
                ).compute_effect(self.data)
                self.assertEqual(got.keys(), want.keys())
                for key in want:
                    self.assertAlmostEqual(got[key], want[key], places=12, msg=key)


if __name__ == "__main__":
    unittest.main()
