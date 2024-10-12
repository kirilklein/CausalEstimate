import unittest
import pandas as pd
from CausalEstimate.stats.utils import dataframe_to_nested_dict


class TestUtils(unittest.TestCase):
    def test_dataframe_to_nested_dict(self):
        # Create a sample DataFrame
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}, index=["X", "Y", "Z"]
        )

        # Convert DataFrame to nested dict
        result = dataframe_to_nested_dict(df)

        # Expected output
        expected = {
            "X": {"A": 1, "B": 4, "C": 7},
            "Y": {"A": 2, "B": 5, "C": 8},
            "Z": {"A": 3, "B": 6, "C": 9},
        }

        # Check if the result matches the expected output
        self.assertEqual(result, expected)

    def test_dataframe_to_nested_dict_empty(self):
        # Test with an empty DataFrame
        df = pd.DataFrame()
        result = dataframe_to_nested_dict(df)
        self.assertEqual(result, {})

    def test_dataframe_to_nested_dict_single_row(self):
        # Test with a DataFrame containing a single row
        df = pd.DataFrame({"A": [1], "B": [2]}, index=["X"])
        result = dataframe_to_nested_dict(df)
        expected = {"X": {"A": 1, "B": 2}}
        self.assertEqual(result, expected)

    def test_dataframe_to_nested_dict_single_column(self):
        # Test with a DataFrame containing a single column
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["X", "Y", "Z"])
        result = dataframe_to_nested_dict(df)
        expected = {"X": {"A": 1}, "Y": {"A": 2}, "Z": {"A": 3}}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
