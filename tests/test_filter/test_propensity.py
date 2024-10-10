import unittest
import pandas as pd
from CausalEstimate.filter.propensity import filter_common_support


class TestCommonSupport(unittest.TestCase):
    def setUp(self):
        self.data = {
            "PID": [1, 2, 3, 4, 5, 6, 7, 8],
            "propensity_score": [0.1, 0.2, 0.9, 0.8, 0.5, 0.3, 0.7, 0.6],
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
        }
        self.df = pd.DataFrame(self.data)

    def test_common_support(self):
        # Run the common support function
        filtered_df = filter_common_support(self.df)
        # Expected filtered DataFrame should contain individuals within the common support range
        expected_pids = [5, 8]  # Based on the calculated common support range
        self.assertListEqual(list(filtered_df["PID"]), expected_pids)

    def test_no_common_support(self):
        # Modify data to have no common support between treated and control groups
        no_overlap_data = {
            "PID": [1, 2, 3, 4, 5, 6],
            "propensity_score": [0.9, 0.95, 0.99, 0.1, 0.05, 0.01],
            "treatment": [1, 1, 1, 0, 0, 0],
        }
        df_no_overlap = pd.DataFrame(no_overlap_data)

        # Run the common support function
        filtered_df = filter_common_support(df_no_overlap)

        # Expect no individuals to be within common support range
        self.assertTrue(filtered_df.empty)

    def test_threshold(self):
        # Test with a higher threshold value to see if individuals are correctly filtered
        filtered_df = filter_common_support(self.df, threshold=0)
        expected_pids = [5, 6, 7, 8]  # More trimming due to a higher threshold
        self.assertListEqual(list(filtered_df["PID"]), expected_pids)


if __name__ == "__main__":
    unittest.main()
