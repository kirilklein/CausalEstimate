import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from CausalEstimate import run_diagnostics
from CausalEstimate.diagnostics import (
    check_balance,
    compute_balance_table,
    compute_positivity_metrics,
    compute_weight_diagnostics,
)
from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL


class TestRunDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alpha = [0.1, 0.6, -0.5]
        data = simulate_binary_data(
            5000, alpha=alpha, beta=[0.5, 0.8, -0.6, 0.3], seed=41
        )
        ps = expit(alpha[0] + alpha[1] * data["X1"] + alpha[2] * data["X2"])
        cls.df = pd.DataFrame(
            {
                TREATMENT_COL: data[TREATMENT_COL].to_numpy(),
                PS_COL: np.clip(ps, 1e-7, 1 - 1e-7),
                "X1": data["X1"].to_numpy(),
                "X2": data["X2"].to_numpy(),
            }
        )

    def test_structure_with_covariates(self):
        report = run_diagnostics(self.df, covariate_cols=["X1", "X2"])
        self.assertEqual(
            set(report),
            {"positivity", "weights", "balance", "balance_summary", "flags"},
        )
        self.assertIsInstance(report["balance"], pd.DataFrame)
        self.assertIsInstance(report["balance_summary"], dict)
        self.assertIsInstance(report["flags"]["extreme_ps"], bool)
        self.assertIsInstance(report["flags"]["unbalanced"], bool)

    def test_structure_without_covariates(self):
        report = run_diagnostics(self.df)
        self.assertIsNone(report["balance"])
        self.assertIsNone(report["balance_summary"])
        self.assertIsNone(report["flags"]["unbalanced"])

    def test_agrees_with_components(self):
        report = run_diagnostics(self.df, covariate_cols=["X1"])
        self.assertEqual(report["positivity"], compute_positivity_metrics(self.df))
        self.assertEqual(report["weights"], compute_weight_diagnostics(self.df))
        expected_table = compute_balance_table(self.df, ["X1"])
        pd.testing.assert_frame_equal(report["balance"], expected_table)
        self.assertEqual(report["balance_summary"], check_balance(expected_table))

    def test_flags_surface(self):
        extreme = pd.DataFrame(
            {
                TREATMENT_COL: [1, 1, 0, 0],
                PS_COL: [0.005, 0.5, 0.5, 0.5],
            }
        )
        report = run_diagnostics(extreme)
        self.assertTrue(report["flags"]["extreme_ps"])

    def test_smd_threshold_passthrough(self):
        strict = run_diagnostics(
            self.df, covariate_cols=["X1", "X2"], smd_threshold=1e-9
        )
        self.assertTrue(strict["flags"]["unbalanced"])
        self.assertFalse(strict["balance"]["balanced"].any())


if __name__ == "__main__":
    unittest.main()
