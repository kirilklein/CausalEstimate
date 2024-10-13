import unittest
import pandas as pd
from CausalEstimate.matching.matching import match_optimal
from CausalEstimate.estimators.functional.matching import compute_matching_ate


class TestMatchingEstimator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "PID": [101, 102, 103, 202, 203, 204, 205, 206],
                "treatment": [1, 1, 1, 0, 0, 0, 0, 0],
                "ps": [0.3, 0.5, 0.7, 0.31, 0.51, 0.71, 0.32, 0.52],
                "outcome": [10, 20, 30, 15, 25, 35, 18, 28],
            }
        )
        self.matching_result = match_optimal(self.df)

    def test_compute_matching_ate_basic(self):
        Y = pd.Series(self.df["outcome"].values, index=self.df["PID"])
        ate = compute_matching_ate(Y, self.matching_result)
        self.assertIsInstance(ate, float)
        self.assertTrue(
            -20 < ate < 20
        )  # Assuming the effect is within a reasonable range

    def test_compute_matching_ate_custom_columns(self):
        Y = pd.Series(self.df["outcome"].values, index=self.df["PID"])
        matching_df = self.matching_result.rename(
            columns={"treated_pid": "treated", "control_pid": "control"}
        )
        ate = compute_matching_ate(
            Y, matching_df, treated_col="treated", control_col="control"
        )
        self.assertIsInstance(ate, float)

    def test_compute_matching_ate_missing_column(self):
        Y = pd.Series(self.df["outcome"].values, index=self.df["PID"])
        matching_df = self.matching_result.drop("control_pid", axis=1)
        with self.assertRaises(ValueError):
            compute_matching_ate(Y, matching_df)

    def test_compute_matching_ate_known_effect(self):
        df = pd.DataFrame(
            {
                "PID": [1, 2, 3, 4],
                "treatment": [1, 1, 0, 0],
                "ps": [0.4, 0.6, 0.41, 0.61],
                "outcome": [10, 20, 5, 15],
            }
        )
        Y = pd.Series(df["outcome"].values, index=df["PID"])
        matching_result = match_optimal(df)
        ate = compute_matching_ate(Y, matching_result)
        self.assertEqual(ate, 5)  # (10-5 + 20-15) / 2 = 5


if __name__ == "__main__":
    unittest.main()
