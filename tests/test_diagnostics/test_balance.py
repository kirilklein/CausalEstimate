import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from CausalEstimate.diagnostics import (
    check_balance,
    compute_balance_table,
    compute_smd,
)
from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL


class TestComputeSMD(unittest.TestCase):
    def test_equal_means_give_zero(self):
        x = np.array([1.0, 2.0, 1.0, 2.0])
        A = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(compute_smd(x, A), 0.0, places=9)

    def test_known_shift_exact_value(self):
        # treated [2, 4], control [1, 3]: mean diff 1, pooled var (2+2)/2 = 2
        x = np.array([2.0, 4.0, 1.0, 3.0])
        A = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(compute_smd(x, A), 1 / np.sqrt(2), places=9)

    def test_uniform_weights_match_unweighted(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=100)
        A = np.array([1] * 50 + [0] * 50)
        w = np.full(100, 2.5)
        self.assertAlmostEqual(compute_smd(x, A), compute_smd(x, A, w), places=9)

    def test_zero_variance_warns_and_returns_nan(self):
        x = np.full(6, 5.0)
        A = np.array([1, 1, 1, 0, 0, 0])
        with self.assertWarns(RuntimeWarning):
            result = compute_smd(x, A)
        self.assertTrue(np.isnan(result))

    def test_exact_att_weighted_value(self):
        # ATT weights: treated 1, control ps/(1-ps) -> [1/3, 1]
        # weighted control mean 2.5, treated mean 3, pooled SD sqrt(2)
        x = np.array([2.0, 4.0, 1.0, 3.0])
        A = np.array([1, 1, 0, 0])
        w = np.array([1.0, 1.0, 1 / 3, 1.0])
        self.assertAlmostEqual(compute_smd(x, A, w), 0.5 / np.sqrt(2), places=9)

    def test_invalid_inputs_raise(self):
        x = np.array([2.0, 4.0, 1.0, 3.0])
        A = np.array([1, 1, 0, 0])
        with self.assertRaises(ValueError):
            compute_smd(x, np.array([1, 0, 0]))  # mismatched shapes
        with self.assertRaises(ValueError):
            compute_smd(x, np.array([1, 2, 0, 0]))  # nonbinary treatment
        with self.assertRaises(ValueError):
            compute_smd(np.array([1.0, np.nan, 2.0, 3.0]), A)
        with self.assertRaises(ValueError):
            compute_smd(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 0]))  # 1-obs arm
        with self.assertRaises(ValueError):
            compute_smd(x, A, weights=np.array([1.0, 1.0, 0.0, 0.0]))  # zero-sum arm
        with self.assertRaises(ValueError):
            compute_smd(x, A, weights=np.array([1.0, -1.0, 1.0, 1.0]))


class TestBalanceTable(unittest.TestCase):
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

    def test_weighting_improves_balance_on_confounders(self):
        table = compute_balance_table(self.df, ["X1", "X2"])
        for cov in ("X1", "X2"):
            self.assertLess(
                abs(table.loc[cov, "smd_weighted"]),
                abs(table.loc[cov, "smd_unweighted"]),
            )

    def test_table_structure(self):
        table = compute_balance_table(self.df, ["X1", "X2"])
        self.assertEqual(list(table.index), ["X1", "X2"])
        self.assertEqual(
            list(table.columns),
            [
                "mean_treated",
                "mean_control",
                "weighted_mean_treated",
                "weighted_mean_control",
                "smd_unweighted",
                "smd_weighted",
                "balanced",
            ],
        )
        self.assertTrue(table["balanced"].dtype == bool)

    def test_att_path_runs(self):
        table = compute_balance_table(self.df, ["X1"], weight_type="ATT")
        self.assertFalse(np.isnan(table.loc["X1", "smd_weighted"]))

    def test_check_balance_summary(self):
        table = compute_balance_table(self.df, ["X1", "X2"], threshold=0.1)
        summary = check_balance(table)
        self.assertEqual(
            summary["balanced"], bool((table["smd_weighted"].abs() < 0.1).all())
        )
        self.assertEqual(
            summary["n_unbalanced"], int((table["smd_weighted"].abs() >= 0.1).sum())
        )
        self.assertAlmostEqual(summary["prop_unbalanced"], summary["n_unbalanced"] / 2)
        self.assertGreaterEqual(summary["max_smd_weighted"], 0.0)

    def test_check_balance_respects_table_threshold(self):
        # summary must agree with the threshold the table was built with
        strict = compute_balance_table(self.df, ["X1", "X2"], threshold=1e-6)
        summary = check_balance(strict)
        self.assertEqual(summary["n_unbalanced"], int((~strict["balanced"]).sum()))
        self.assertFalse(summary["balanced"])

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            compute_balance_table(self.df, ["missing_col"])
        nan_df = self.df.copy()
        nan_df.loc[nan_df.index[0], "X1"] = np.nan
        with self.assertRaises(ValueError):
            compute_balance_table(nan_df, ["X1"])
        with self.assertRaises(ValueError):
            compute_balance_table(self.df, [])
        with self.assertRaises(ValueError):
            compute_balance_table(self.df, ["X1"], threshold=np.nan)
        with self.assertRaises(ValueError):
            check_balance(pd.DataFrame({"smd_weighted": []}))

    def test_undefined_smd_never_certified_balanced(self):
        df = self.df.copy()
        df["const"] = 1.0
        with self.assertWarns(RuntimeWarning):
            table = compute_balance_table(df, ["X1", "const"])
        self.assertFalse(table.loc["const", "balanced"])
        self.assertTrue(np.isnan(table.loc["const", "smd_weighted"]))
        summary = check_balance(table)
        self.assertEqual(summary["n_undefined"], 1)
        self.assertFalse(summary["balanced"])
        # proportions are over evaluable covariates only (here: X1)
        self.assertAlmostEqual(summary["prop_unbalanced"], summary["n_unbalanced"] / 1)


if __name__ == "__main__":
    unittest.main()
