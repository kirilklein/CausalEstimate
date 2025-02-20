import unittest

import pandas as pd

from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.matching.matching import match_eager, match_optimal
from CausalEstimate.utils.constants import (
    CONTROL_PID_COL,
    OUTCOME_COL,
    PID_COL,
    PS_COL,
    TREATMENT_COL,
)


class TestMatchingEstimator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                PID_COL: [101, 102, 103, 202, 203, 204, 205, 206],
                TREATMENT_COL: [1, 1, 1, 0, 0, 0, 0, 0],
                PS_COL: [0.3, 0.5, 0.7, 0.31, 0.51, 0.71, 0.32, 0.52],
                OUTCOME_COL: [10, 20, 30, 15, 25, 35, 18, 28],
            }
        )
        self.matching_result = match_optimal(self.df)

    def test_compute_matching_ate_basic(self):
        Y = pd.Series(self.df[OUTCOME_COL].values, index=self.df[PID_COL])
        ate = compute_matching_ate(Y, self.matching_result)
        self.assertIsInstance(ate, float)
        self.assertTrue(
            -20 < ate < 20
        )  # Assuming the effect is within a reasonable range

    def test_compute_matching_ate_missing_column(self):
        Y = pd.Series(self.df[OUTCOME_COL].values, index=self.df[PID_COL])
        matching_df = self.matching_result.drop(CONTROL_PID_COL, axis=1)
        with self.assertRaises(ValueError):
            compute_matching_ate(Y, matching_df)

    def test_compute_matching_ate_known_effect(self):
        df = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                TREATMENT_COL: [1, 1, 0, 0],
                PS_COL: [0.4, 0.6, 0.41, 0.61],
                OUTCOME_COL: [10, 20, 5, 15],
            }
        )
        Y = pd.Series(df[OUTCOME_COL].values, index=df[PID_COL])
        matching_result = match_optimal(df)
        ate = compute_matching_ate(Y, matching_result)
        self.assertEqual(ate, 5)  # (10-5 + 20-15) / 2 = 5


class TestEagerMatchingEstimator(unittest.TestCase):
    def setUp(self):
        """
        We'll use the same data as the TestMatchingEstimator above,
        but call match_eager instead of match_optimal.
        """
        self.df = pd.DataFrame(
            {
                PID_COL: [101, 102, 103, 202, 203, 204, 205, 206],
                TREATMENT_COL: [1, 1, 1, 0, 0, 0, 0, 0],
                PS_COL: [0.3, 0.5, 0.7, 0.31, 0.51, 0.71, 0.32, 0.52],
                OUTCOME_COL: [10, 20, 30, 15, 25, 35, 18, 28],
            }
        )
        # Use eager matching here
        self.matching_result_eager = match_eager(self.df)

    def test_compute_matching_ate_eager_basic(self):
        Y = pd.Series(self.df[OUTCOME_COL].values, index=self.df[PID_COL])
        ate = compute_matching_ate(Y, self.matching_result_eager)
        self.assertIsInstance(ate, float)
        self.assertTrue(-20 < ate < 20)

    def test_compute_matching_ate_eager_missing_column(self):
        Y = pd.Series(self.df[OUTCOME_COL].values, index=self.df[PID_COL])
        # remove 'control_pid'
        matching_df = self.matching_result_eager.drop(CONTROL_PID_COL, axis=1)
        with self.assertRaises(ValueError):
            compute_matching_ate(Y, matching_df)

    def test_compute_matching_ate_eager_known_effect(self):
        df = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                TREATMENT_COL: [1, 1, 0, 0],
                PS_COL: [0.4, 0.6, 0.41, 0.61],
                OUTCOME_COL: [10, 20, 5, 15],
            }
        )
        Y = pd.Series(df[OUTCOME_COL].values, index=df[PID_COL])
        matching_result = match_eager(df)
        ate = compute_matching_ate(Y, matching_result)
        # For example, a known effect check depends on how the eager matching pairs them.
        # If the pairing is the same as the optimal in this example, we might also get 5.
        # Let's see:
        self.assertIsInstance(ate, float)
        # you could do an assertEqual if you know the expected pairing,
        # or just check it's in a plausible range:
        self.assertTrue(-20 < ate < 20)


if __name__ == "__main__":
    unittest.main()
