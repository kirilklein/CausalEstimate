import unittest
import pandas as pd
import numpy as np
from CausalEstimate.utils.utils import (
    get_treated_ps,
    get_untreated_ps,
    get_treated,
    get_untreated,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, n),
                "ps": np.random.uniform(0, 1, n),
                "outcome": np.random.normal(0, 1, n),
            }
        )

    def test_get_treated(self):
        treated = get_treated(self.df, "treatment")
        self.assertTrue(all(treated["treatment"] == 1))
        self.assertEqual(len(treated), self.df["treatment"].sum())

    def test_get_untreated(self):
        untreated = get_untreated(self.df, "treatment")
        self.assertTrue(all(untreated["treatment"] == 0))
        self.assertEqual(len(untreated), len(self.df) - self.df["treatment"].sum())

    def test_get_treated_ps(self):
        treated_ps = get_treated_ps(self.df, "treatment", "ps")
        self.assertEqual(len(treated_ps), self.df["treatment"].sum())
        self.assertTrue(
            all(treated_ps.index == self.df[self.df["treatment"] == 1].index)
        )

    def test_get_untreated_ps(self):
        untreated_ps = get_untreated_ps(self.df, "treatment", "ps")
        self.assertEqual(len(untreated_ps), len(self.df) - self.df["treatment"].sum())
        self.assertTrue(
            all(untreated_ps.index == self.df[self.df["treatment"] == 0].index)
        )

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["treatment", "ps", "outcome"])
        self.assertTrue(get_treated(empty_df, "treatment").empty)
        self.assertTrue(get_untreated(empty_df, "treatment").empty)
        self.assertTrue(get_treated_ps(empty_df, "treatment", "ps").empty)
        self.assertTrue(get_untreated_ps(empty_df, "treatment", "ps").empty)

    def test_all_treated(self):
        all_treated_df = pd.DataFrame(
            {
                "treatment": [1] * 10,
                "ps": np.random.uniform(0, 1, 10),
                "outcome": np.random.normal(0, 1, 10),
            }
        )
        self.assertEqual(len(get_treated(all_treated_df, "treatment")), 10)
        self.assertTrue(get_untreated(all_treated_df, "treatment").empty)
        self.assertEqual(len(get_treated_ps(all_treated_df, "treatment", "ps")), 10)
        self.assertTrue(get_untreated_ps(all_treated_df, "treatment", "ps").empty)

    def test_all_untreated(self):
        all_untreated_df = pd.DataFrame(
            {
                "treatment": [0] * 10,
                "ps": np.random.uniform(0, 1, 10),
                "outcome": np.random.normal(0, 1, 10),
            }
        )
        self.assertTrue(get_treated(all_untreated_df, "treatment").empty)
        self.assertEqual(len(get_untreated(all_untreated_df, "treatment")), 10)
        self.assertTrue(get_treated_ps(all_untreated_df, "treatment", "ps").empty)
        self.assertEqual(len(get_untreated_ps(all_untreated_df, "treatment", "ps")), 10)

    def test_missing_columns(self):
        df_missing_treatment = self.df.drop("treatment", axis=1)
        df_missing_ps = self.df.drop("ps", axis=1)

        with self.assertRaises(KeyError):
            get_treated(df_missing_treatment, "treatment")
        with self.assertRaises(KeyError):
            get_untreated(df_missing_treatment, "treatment")
        with self.assertRaises(KeyError):
            get_treated_ps(df_missing_treatment, "treatment", "ps")
        with self.assertRaises(KeyError):
            get_treated_ps(df_missing_ps, "treatment", "ps")


if __name__ == "__main__":
    unittest.main()
