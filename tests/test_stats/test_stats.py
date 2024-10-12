import unittest

import numpy as np
import pandas as pd

from CausalEstimate.stats.stats import compute_treatment_outcome_table


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


if __name__ == "__main__":
    unittest.main()
