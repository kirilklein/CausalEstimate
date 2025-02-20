import unittest

import numpy as np
import pandas as pd

from CausalEstimate.utils.constants import OUTCOME_COL, PS_COL, TREATMENT_COL
from CausalEstimate.utils.utils import (
    get_treated,
    get_treated_ps,
    get_untreated,
    get_untreated_ps,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                TREATMENT_COL: np.random.binomial(1, 0.5, n),
                PS_COL: np.random.uniform(0, 1, n),
                OUTCOME_COL: np.random.normal(0, 1, n),
            }
        )

    def test_get_treated(self):
        treated = get_treated(self.df, TREATMENT_COL)
        self.assertTrue(all(treated[TREATMENT_COL] == 1))
        self.assertEqual(len(treated), self.df[TREATMENT_COL].sum())

    def test_get_untreated(self):
        untreated = get_untreated(self.df, TREATMENT_COL)
        self.assertTrue(all(untreated[TREATMENT_COL] == 0))
        self.assertEqual(len(untreated), len(self.df) - self.df[TREATMENT_COL].sum())

    def test_get_treated_ps(self):
        treated_ps = get_treated_ps(self.df, TREATMENT_COL, PS_COL)
        self.assertEqual(len(treated_ps), self.df[TREATMENT_COL].sum())
        self.assertTrue(
            all(treated_ps.index == self.df[self.df[TREATMENT_COL] == 1].index)
        )

    def test_get_untreated_ps(self):
        untreated_ps = get_untreated_ps(self.df, TREATMENT_COL, PS_COL)
        self.assertEqual(len(untreated_ps), len(self.df) - self.df[TREATMENT_COL].sum())
        self.assertTrue(
            all(untreated_ps.index == self.df[self.df[TREATMENT_COL] == 0].index)
        )

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=[TREATMENT_COL, PS_COL, OUTCOME_COL])
        self.assertTrue(get_treated(empty_df, TREATMENT_COL).empty)
        self.assertTrue(get_untreated(empty_df, TREATMENT_COL).empty)
        self.assertTrue(get_treated_ps(empty_df, TREATMENT_COL, PS_COL).empty)
        self.assertTrue(get_untreated_ps(empty_df, TREATMENT_COL, PS_COL).empty)

    def test_all_treated(self):
        all_treated_df = pd.DataFrame(
            {
                TREATMENT_COL: [1] * 10,
                PS_COL: np.random.uniform(0, 1, 10),
                OUTCOME_COL: np.random.normal(0, 1, 10),
            }
        )
        self.assertEqual(len(get_treated(all_treated_df, TREATMENT_COL)), 10)
        self.assertTrue(get_untreated(all_treated_df, TREATMENT_COL).empty)
        self.assertEqual(len(get_treated_ps(all_treated_df, TREATMENT_COL, PS_COL)), 10)
        self.assertTrue(get_untreated_ps(all_treated_df, TREATMENT_COL, PS_COL).empty)

    def test_all_untreated(self):
        all_untreated_df = pd.DataFrame(
            {
                TREATMENT_COL: [0] * 10,
                PS_COL: np.random.uniform(0, 1, 10),
                OUTCOME_COL: np.random.normal(0, 1, 10),
            }
        )
        self.assertTrue(get_treated(all_untreated_df, TREATMENT_COL).empty)
        self.assertEqual(len(get_untreated(all_untreated_df, TREATMENT_COL)), 10)
        self.assertTrue(get_treated_ps(all_untreated_df, TREATMENT_COL, PS_COL).empty)
        self.assertEqual(
            len(get_untreated_ps(all_untreated_df, TREATMENT_COL, PS_COL)), 10
        )

    def test_missing_columns(self):
        df_missing_treatment = self.df.drop(TREATMENT_COL, axis=1)
        df_missing_ps = self.df.drop(PS_COL, axis=1)

        with self.assertRaises(KeyError):
            get_treated(df_missing_treatment, TREATMENT_COL)
        with self.assertRaises(KeyError):
            get_untreated(df_missing_treatment, TREATMENT_COL)
        with self.assertRaises(KeyError):
            get_treated_ps(df_missing_treatment, TREATMENT_COL, PS_COL)
        with self.assertRaises(KeyError):
            get_treated_ps(df_missing_ps, TREATMENT_COL, PS_COL)


if __name__ == "__main__":
    unittest.main()
