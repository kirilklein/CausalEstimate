import unittest

import numpy as np
import pandas as pd

from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.matching.matching import match_optimal
from tests.helpers.setup import TestEffectBase


class TestMatching(unittest.TestCase):
    """
    Basic example test for matching
    """

    def setUp(self):
        # Create a sample DataFrame for testing
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

    def test_match_optimal_basic(self):
        result = match_optimal(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(
            list(result.columns), ["treated_pid", "control_pid", "distance"]
        )
        self.assertEqual(len(result), sum(self.df["treatment"] == 1))

    def test_match_optimal_n_controls(self):
        n_controls = 2
        result = match_optimal(self.df, n_controls=n_controls)
        self.assertEqual(len(result), sum(self.df["treatment"] == 1) * n_controls)

    def test_match_optimal_caliper(self):
        caliper = 0.1
        result = match_optimal(self.df, caliper=caliper)
        self.assertTrue(all(result["distance"] <= caliper))

    def test_match_optimal_custom_columns(self):
        df = self.df.rename(
            columns={"treatment": "treat", "ps": "propensity", "PID": "ID"}
        )
        result = match_optimal(
            df, treatment_col="treat", ps_col="propensity", pid_col="ID"
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(
            list(result.columns), ["treated_pid", "control_pid", "distance"]
        )

    def test_match_optimal_insufficient_controls(self):
        df = pd.DataFrame(
            {
                "PID": range(10),
                "treatment": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "ps": np.linspace(0, 1, 10),
            }
        )
        with self.assertRaises(ValueError):
            match_optimal(df, n_controls=2)

    def test_match_optimal_all_treated(self):
        df = self.df.copy()
        df["treatment"] = 1
        with self.assertRaises(ValueError):
            match_optimal(df)


class BaseTestComputeMatchingATE(TestEffectBase):
    alpha = [-1, 0.1, 0.1, 0]

    def test_compute_matching_ate(self):
        ate_matching = compute_matching_ate(self.data["Y"], match_optimal(self.data))
        self.assertAlmostEqual(ate_matching, self.true_ate, delta=0.1)


class TestComputeMatchingATE_ps_interaction(BaseTestComputeMatchingATE):
    alpha = [-1, 0.1, 0.1, 2]


class TestComputeMatchingATE_outcome_interaction(BaseTestComputeMatchingATE):
    beta = [-1, 0.1, 0.1, 2, 2]


class TestComputeMatchingATE_ps_and_outcome_interaction(BaseTestComputeMatchingATE):
    """This one we expect to not be estimable"""

    alpha = [-1, 0.1, 0.1, 2]
    beta = [-1, 0.1, 0.1, 2, 2]

    def test_compute_matching_ate(self):
        ate_matching = compute_matching_ate(self.data["Y"], match_optimal(self.data))
        self.assertNotAlmostEqual(ate_matching, self.true_ate, delta=0.05)


if __name__ == "__main__":
    unittest.main()
