import unittest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from CausalEstimate.matching.matching import (
    match_optimal,
)  # Replace with the correct import path


class TestMatchOptimal(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame to use in the tests."""
        # DataFrame setup based on the provided description
        self.df = pd.DataFrame(
            {
                "PID": [
                    101,
                    102,
                    103,
                    202,
                    203,
                    204,
                    205,
                    206,
                    207,
                    208,
                    209,
                    210,
                    211,
                ],
                "treatment": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ps": [
                    0.3,
                    0.90,
                    0.5,
                    0.31,
                    0.32,
                    0.33,
                    0.36,
                    0.91,
                    0.92,
                    0.93,
                    0.94,
                    0.49,
                    0.52,
                ],  # Unique propensity scores
            }
        )

    def sort_dataframe(self, df):
        """Helper function to sort DataFrame by 'treated_pid' and 'control_pid'."""
        return df.sort_values(by=["treated_pid", "control_pid"]).reset_index(drop=True)

    def test_1_to_2_matching_with_caliper(self):
        """Test 1:2 matching with a caliper of 0.05 for all treated units."""
        result = match_optimal(self.df, n_controls=2, caliper=0.05)

        # Expected matching:
        # - Treated 101 (ps=0.3) will match with Controls
        #  202 (ps=0.31), 203 (ps=0.32) -> distances: 0.01, 0.02
        #
        # - Treated 102 (ps=0.90) will match with Controls
        # 206 (ps=0.91), 207 (ps=0.92) -> distances: 0.01, 0.02
        #
        # - Treated 103 (ps=0.5) will match with Controls
        # 210 (ps=0.49), 211 (ps=0.52) -> distances: 0.01, 0.02
        expected = pd.DataFrame(
            {
                "treated_pid": [101, 101, 102, 102, 103, 103],
                "control_pid": [202, 203, 206, 207, 210, 211],
                "distance": [0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            }
        )

        # Sort both DataFrames before comparison
        result_sorted = self.sort_dataframe(
            result[["treated_pid", "control_pid", "distance"]]
        )
        expected_sorted = self.sort_dataframe(expected)

        pd.testing.assert_frame_equal(result_sorted, expected_sorted)

    def test_1_to_3_matching_with_caliper(self):
        """
        Test 1:3 matching, but Treated 103 will be dropped as it can't find
        3 matches.
        """
        result = match_optimal(self.df, n_controls=3, caliper=0.05)

        # Expected matching:
        # - Treated 101 (ps=0.3) will match with Controls
        # 202 (ps=0.31), 203 (ps=0.32), 204 (ps=0.33)
        # -> distances: 0.01, 0.02, 0.03
        #
        # - Treated 102 (ps=0.90) will match with Controls
        # 206 (ps=0.91), 207 (ps=0.92), 208 (ps=0.93)
        # -> distances: 0.01, 0.02, 0.03
        #
        # - Treated 103 (ps=0.5) will be dropped because
        # it can't find 3 matches within the caliper.
        expected = pd.DataFrame(
            {
                "treated_pid": [101, 101, 101, 102, 102, 102],
                "control_pid": [202, 203, 204, 206, 207, 208],
                "distance": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
            }
        )

        # Sort both DataFrames before comparison
        result_sorted = self.sort_dataframe(
            result[["treated_pid", "control_pid", "distance"]]
        )
        expected_sorted = self.sort_dataframe(expected)

        pd.testing.assert_frame_equal(result_sorted, expected_sorted)

    def test_treated_unit_with_insufficient_controls_due_to_caliper(self):
        """
        Test treated units where matching is restricted by the caliper and
        no matches are possible."""
        # Expecting a ValueError because no treated unit has enough controls
        # within the caliper of 0.02
        with self.assertRaises(
            ValueError, msg="No treated units have sufficient controls"
        ):
            match_optimal(self.df, n_controls=3, caliper=0.02)

    def test_matching_with_no_treated_units(self):
        """
        Test case where no treated individuals are present.
        """
        df_no_treated = self.df[
            self.df["treatment"] == 0
        ].copy()  # Remove treated individuals
        with self.assertRaises(
            ValueError, msg="No treated units have sufficient controls"
        ):
            match_optimal(df_no_treated, n_controls=1)

    def test_insufficient_controls(self):
        """
        Test case where there aren't enough controls to
        match the treated units.
        """
        df_insufficient_controls = pd.DataFrame(
            {
                "PID": [101, 102],
                "treatment": [1, 1],
                "ps": [0.9, 0.85],  # Only treated individuals, no controls
            }
        )
        with self.assertRaises(ValueError, msg="Not enough controls to match"):
            match_optimal(df_insufficient_controls, n_controls=1)


class TestMatchOptimalWithStats(unittest.TestCase):

    def setUp(self):
        """
        Simulate 100 treated and 1000 control units with different
        propensity score distributions.
        """
        np.random.seed(42)  # For reproducibility

        # Simulate 100 treated units with ps around 0.7
        treated_ps = np.random.beta(2, 3, size=100)

        # Simulate 1000 control units with ps around 0.5
        # (more spread out to create imbalance)
        control_ps = np.random.beta(3, 2, size=1000)

        # Combine treated and control units into a DataFrame
        self.df = pd.DataFrame(
            {
                "PID": np.arange(1, 1101),
                "treatment": np.concatenate([np.ones(100), np.zeros(1000)]),
                "ps": np.concatenate([treated_ps, control_ps]),
            }
        )

    def test_ks_stat_before_and_after_matching(self):
        """
        Test if the propensity score distributions are balanced
        after matching using KS test.
        """
        # Step 1: Perform matching (1:1 matching with a caliper of 0.2)
        result = match_optimal(self.df, n_controls=1, caliper=0.2)

        # Step 2: Extract propensity scores before and after matching
        treated_ps = self.df[self.df["treatment"] == 1]["ps"].values
        control_ps = self.df[self.df["treatment"] == 0]["ps"].values
        matched_control_ps = self.df[self.df["PID"].isin(result["control_pid"])][
            "ps"
        ].values

        # Step 3: Perform KS test before matching (treated vs all controls)
        ks_before = ks_2samp(treated_ps, control_ps)
        print(
            f"KS test before matching: \
              statistic={ks_before.statistic}, p-value={ks_before.pvalue}"
        )

        # Step 4: Perform KS test after matching (treated vs matched controls)
        ks_after = ks_2samp(treated_ps, matched_control_ps)
        print(
            f"KS test after matching:\
               statistic={ks_after.statistic}, p-value={ks_after.pvalue}"
        )

        # Step 5: Assert that the KS test shows a significant difference before matching (p-value < 0.05)
        self.assertLess(
            ks_before.pvalue,
            0.05,
            "Expected significant difference in ps distributions before matching",
        )

        # Step 6: Assert that the KS test shows no significant difference after matching (p-value >= 0.05)
        self.assertGreaterEqual(
            ks_after.pvalue,
            0.05,
            "Expected no significant difference in ps distributions after matching (good balance)",
        )


if __name__ == "__main__":
    unittest.main()
