import unittest

import pandas as pd

from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.utils.constants import CONTROL_PID_COL, TREATED_PID_COL, EFFECT


# Create the unittest class
class TestComputeMatchingATE(unittest.TestCase):
    """Basic tests for matching estimators"""

    def test_compute_matching_ate(self):
        # Example outcome array (Y)
        Y = pd.Series(
            {
                0: 5.0,  # control
                1: 6.5,  # treated
                2: 3.0,  # control
                3: 7.2,  # treated
                4: 4.0,  # control
                5: 5.5,  # treated
                6: 2.0,  # control
                7: 1.0,  # control
                8: 3.8,  # control
                9: 4.7,  # control
            }
        )

        # Example matching DataFrame (1:2 matching)
        matching_df = pd.DataFrame(
            {
                TREATED_PID_COL: [1, 1, 3, 3, 5, 5],  # Treated unit indices
                CONTROL_PID_COL: [0, 2, 4, 6, 7, 8],  # Matched control unit indices
            }
        )

        # Expected ATE for the matched data
        expected_ate = (
            6.5 - (5.0 + 3.0) / 2 + 7.2 - (4.0 + 2.0) / 2 + 5.5 - (1.0 + 3.8) / 2
        ) / 3

        # Run the function
        result_ate = compute_matching_ate(Y, matching_df)

        # Check if the result matches the expected ATE
        self.assertAlmostEqual(result_ate[EFFECT], expected_ate, places=5)

    def test_missing_columns(self):
        # Test for missing columns in the matching DataFrame
        Y = pd.Series({0: 5.0, 1: 6.5})

        # Example DataFrame missing the 'control_pid' column
        matching_df = pd.DataFrame({TREATED_PID_COL: [1]})

        # Ensure that a ValueError is raised for missing columns
        with self.assertRaises(ValueError):
            compute_matching_ate(Y, matching_df)


# Run the unittests
unittest.main(argv=[""], exit=False)
