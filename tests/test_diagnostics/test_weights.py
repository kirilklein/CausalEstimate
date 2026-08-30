import unittest

import numpy as np
import pandas as pd

from CausalEstimate.diagnostics import compute_ess, compute_weight_diagnostics
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL
from tests.helpers.setup import TestEffectBase


class TestComputeESS(unittest.TestCase):
    def test_uniform_weights_give_n(self):
        self.assertEqual(compute_ess(np.ones(50)), 50.0)
        self.assertAlmostEqual(compute_ess(np.full(50, 3.7)), 50.0, places=9)

    def test_dominant_weight_approaches_one(self):
        w = np.array([1e6] + [1e-6] * 99)
        self.assertAlmostEqual(compute_ess(w), 1.0, places=6)

    def test_two_value_closed_form(self):
        w = np.array([1.0, 3.0])
        self.assertAlmostEqual(compute_ess(w), 16 / 10)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_ess(np.array([]))


class TestWeightDiagnostics(TestEffectBase):
    def test_ate_diagnostics_structure_and_bounds(self):
        diag = compute_weight_diagnostics(self.data, weight_type="ATE")
        self.assertEqual(diag["n_total"], self.n)
        self.assertLessEqual(diag["ess_treated"], diag["n_treated"])
        self.assertLessEqual(diag["ess_control"], diag["n_control"])
        self.assertLessEqual(diag["ess_total"], diag["n_total"])
        self.assertGreater(diag["ess_fraction_total"], 0.0)
        self.assertLessEqual(diag["ess_fraction_total"], 1.0)
        self.assertLessEqual(diag["weight_q95"], diag["weight_q99"])
        self.assertLessEqual(diag["weight_q99"], diag["max_weight"])

    def test_att_treated_weights_are_one(self):
        diag = compute_weight_diagnostics(self.data, weight_type="ATT")
        self.assertEqual(diag["ess_treated"], diag["n_treated"])
        self.assertEqual(diag["max_weight_treated"], 1.0)

    def test_clipping_raises_ess(self):
        raw = compute_weight_diagnostics(self.data, weight_type="ATE")
        clipped = compute_weight_diagnostics(
            self.data, weight_type="ATE", clip_percentile=0.95
        )
        self.assertGreater(clipped["ess_total"], raw["ess_total"])
        self.assertLess(clipped["max_weight"], raw["max_weight"])


class TestWeightDiagnosticsSmall(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                TREATMENT_COL: [1, 1, 0, 0],
                PS_COL: [0.5, 0.5, 0.5, 0.5],
            }
        )

    def test_constant_ps_gives_full_ess(self):
        diag = compute_weight_diagnostics(self.df, weight_type="ATE")
        self.assertAlmostEqual(diag["ess_total"], 4.0)
        self.assertAlmostEqual(diag["mean_weight"], 2.0, places=6)

    def test_custom_column_names(self):
        renamed = self.df.rename(columns={TREATMENT_COL: "arm", PS_COL: "score"})
        diag = compute_weight_diagnostics(renamed, ps_col="score", treatment_col="arm")
        self.assertAlmostEqual(diag["ess_total"], 4.0)

    def test_single_arm_raises(self):
        with self.assertRaises(ValueError):
            compute_weight_diagnostics(self.df[self.df[TREATMENT_COL] == 1])


if __name__ == "__main__":
    unittest.main()
