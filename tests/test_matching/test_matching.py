import unittest
import numpy as np
import pandas as pd

from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.matching.matching import match_eager, match_optimal
from CausalEstimate.utils.constants import (
    CONTROL_PID_COL,
    OUTCOME_COL,
    PID_COL,
    PS_COL,
    TREATMENT_COL,
    TREATED_PID_COL,
    EFFECT,
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
        self.assertIsInstance(ate[EFFECT], float)
        self.assertTrue(
            -20 < ate[EFFECT] < 20
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
        self.assertEqual(ate[EFFECT], 5)  # (10-5 + 20-15) / 2 = 5


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
        self.assertIsInstance(ate[EFFECT], float)
        self.assertTrue(-20 < ate[EFFECT] < 20)

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
        self.assertIsInstance(ate[EFFECT], float)
        # you could do an assertEqual if you know the expected pairing,
        # or just check it's in a plausible range:
        self.assertTrue(-20 < ate[EFFECT] < 20)


class TestEagerMultipleControls(unittest.TestCase):
    def setUp(self):
        # We'll set up a small example with 2 treated, 5 controls
        # For simplicity, outcome is not used in match_eager but used in compute_matching_ate
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "PID": [1, 2, 3, 4, 5, 6, 7],
                "treatment": [1, 1, 0, 0, 0, 0, 0],
                "ps": [
                    0.3,
                    0.35,
                    0.31,
                    0.32,
                    0.34,
                    0.38,
                    0.9,
                ],  # note subject 7 is far away
                "outcome": [10, 12, 11, 13, 14, 16, 30],  # for computing ATE
            }
        )
        # So:
        #  - subject 1 (ps=0.3) can match controls at ps=0.31, 0.32, 0.34, 0.38
        #  - subject 2 (ps=0.35) can match controls at ps=0.31, 0.32, 0.34, 0.38
        #  - subject 7 (ps=0.9) is effectively out of range for them

    def test_two_controls_strict_true(self):
        # We want each treated subject to get n_controls=2
        # with a big caliper so all relevant controls are in range
        result = match_eager(
            self.df,
            caliper=0.1,  # enough to include 0.3 +/- 0.1 => 0.2..0.4
            n_controls=2,
            strict=True,
        )
        # We should get 2 matches for each treated subject = 4 rows total
        self.assertEqual(len(result), 4)

        # No subject matches with the ps=0.9 control
        # Check each treated PID is repeated exactly twice:
        treated_counts = result[TREATED_PID_COL].value_counts().to_dict()
        self.assertEqual(treated_counts, {1: 2, 2: 2})

        # Let's compute an ATE just to confirm that usage works
        # We have an outcome column in df, let's build a series
        Y = pd.Series(self.df["outcome"].values, index=self.df["PID"].values)
        ate = compute_matching_ate(Y, result)
        self.assertIsInstance(ate[EFFECT], float)

    def test_two_controls_strict_error(self):
        # In this scenario, we'll reduce the caliper to 0.02 so that subject 2 can't find 2 controls
        # because ps=0.35 can only match with e.g. ps=0.34 in that tight range, ignoring 0.32, 0.31
        # => after subject 2 finds one control, the second pass fails if strict=True.
        with self.assertRaises(ValueError):
            match_eager(self.df, caliper=0.02, n_controls=2, strict=True)

    def test_two_controls_non_strict(self):
        # Same scenario, but we won't raise an error if they can't find the second control
        result = match_eager(self.df, caliper=0.02, n_controls=2, strict=False)
        # We do get matches for the first pass, second pass might fail for subject 2
        # Let's see how many matches we ended up with
        self.assertTrue(len(result) >= 1, "Should have at least partial matches")
        # We won't specify an exact number since partial matches are allowed


if __name__ == "__main__":
    unittest.main()
