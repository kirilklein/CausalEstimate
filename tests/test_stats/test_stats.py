import unittest

import numpy as np
import pandas as pd
from scipy import stats

from CausalEstimate.stats.stats import (
    compute_propensity_score_stats,
    compute_treatment_outcome_table,
)


class TestStats(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)
        n = 100
        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.binomial(1, 0.6, n)
        self.df = pd.DataFrame({"treatment": treatment, "outcome": outcome})

    def test_compute_treatment_outcome_table(self):
        # Compute the table
        table = compute_treatment_outcome_table(self.df, "treatment", "outcome")

        # Check that the table has the correct shape
        self.assertEqual(table.shape, (3, 3))

        # Check that the row and column names are correct
        self.assertListEqual(list(table.index), ["Untreated", "Treated", "Total"])
        self.assertListEqual(list(table.columns), ["No Outcome", "Outcome", "Total"])

        # Check that the totals are correct
        self.assertEqual(table.loc["Total", "Total"], len(self.df))
        self.assertEqual(
            table.loc["Untreated", "Total"] + table.loc["Treated", "Total"],
            len(self.df),
        )
        self.assertEqual(
            table.loc["Total", "No Outcome"] + table.loc["Total", "Outcome"],
            len(self.df),
        )

        # Check that the individual cell counts sum up to the totals
        self.assertEqual(
            table.loc["Untreated", "No Outcome"] + table.loc["Untreated", "Outcome"],
            table.loc["Untreated", "Total"],
        )
        self.assertEqual(
            table.loc["Treated", "No Outcome"] + table.loc["Treated", "Outcome"],
            table.loc["Treated", "Total"],
        )

    def test_compare_ps_distributions(self):
        # Create a sample DataFrame with known propensity scores
        np.random.seed(42)
        n = 1000
        treatment = np.random.binomial(1, 0.5, n)

        # Generate propensity scores from different distributions for treated and untreated
        ps_treated = np.random.beta(2, 5, n)
        ps_untreated = np.random.beta(5, 2, n)
        ps = np.where(treatment == 1, ps_treated, ps_untreated)

        df = pd.DataFrame({"treatment": treatment, "ps": ps})

        # Compute the comparison
        result = compute_propensity_score_stats(df, "ps", "treatment")

        # Check that the result contains the expected keys
        self.assertIn("ks_statistic", result)
        self.assertIn("p_value", result)

        # Check that the values are of the correct type
        self.assertIsInstance(result["ks_statistic"], float)
        self.assertIsInstance(result["p_value"], float)

        # Check that the KS statistic is between 0 and 1
        self.assertTrue(0 <= result["ks_statistic"] <= 1)

        # Check that the p-value is between 0 and 1
        self.assertTrue(0 <= result["p_value"] <= 1)

        # As we've used different distributions, we expect a low p-value
        self.assertLess(result["p_value"], 0.05)

        # Compute the KS test directly and compare results
        treated = df[df["treatment"] == 1]["ps"]
        untreated = df[df["treatment"] == 0]["ps"]
        ks_statistic, p_value = stats.ks_2samp(treated, untreated)

        self.assertAlmostEqual(result["ks_statistic"], ks_statistic, places=7)
        self.assertAlmostEqual(result["p_value"], p_value, places=7)


if __name__ == "__main__":
    unittest.main()
