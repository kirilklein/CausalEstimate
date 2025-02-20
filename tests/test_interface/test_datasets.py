import unittest

import numpy as np
import pandas as pd

from CausalEstimate.datasets import load_binary, load_binary_with_probas
from CausalEstimate.utils.constants import (
    TREATMENT_COL,
    OUTCOME_COL,
    PS_COL,
    PROBAS_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
    OUTCOME_CF_COL,
)


class TestBinaryDataset(unittest.TestCase):
    """Test suite for the binary dataset loader"""

    def setUp(self):
        """Set up test fixtures"""
        self.n_samples = 100
        self.random_state = 42

    def test_basic_loading(self):
        """Test basic dataset loading functionality"""
        data = load_binary(n_samples=self.n_samples, random_state=self.random_state)

        # Test type and shape
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (self.n_samples, 5))

        # Test column names
        expected_columns = {"X1", "X2", TREATMENT_COL, OUTCOME_COL, OUTCOME_CF_COL}
        self.assertEqual(set(data.columns), expected_columns)

    def test_binary_values(self):
        """Test that treatment and outcome are binary"""
        data = load_binary(n_samples=self.n_samples)

        # Test treatment values
        treatment_values = set(data[TREATMENT_COL].unique())
        self.assertTrue(treatment_values.issubset({0, 1}))

        # Test outcome values
        outcome_values = set(data[OUTCOME_COL].unique())
        self.assertTrue(outcome_values.issubset({0, 1}))

    def test_reproducibility(self):
        """Test reproducibility with fixed random state"""
        data1 = load_binary(n_samples=50, random_state=42)
        data2 = load_binary(n_samples=50, random_state=42)

        pd.testing.assert_frame_equal(data1, data2)

    def test_parameter_return(self):
        """Test parameter return functionality"""
        data, params = load_binary(n_samples=10, return_params=True)

        # Test params structure
        self.assertIsInstance(params, dict)
        self.assertIn("treatment_params", params)
        self.assertIn("outcome_params", params)
        self.assertIn("DESCR", params)

        # Test description string
        self.assertIsInstance(params["DESCR"], str)
        self.assertGreater(len(params["DESCR"]), 0)

        # Test parameter lists
        self.assertIsInstance(params["treatment_params"], list)
        self.assertIsInstance(params["outcome_params"], list)


class TestBinaryDatasetWithProbas(unittest.TestCase):
    """Test suite for the binary dataset loader with probabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.n_samples = 5000
        self.random_state = 42
        self.expected_columns = {
            "X1",
            "X2",
            TREATMENT_COL,
            OUTCOME_COL,
            PS_COL,
            PROBAS_COL,
            PROBAS_T0_COL,
            PROBAS_T1_COL,
        }

    def test_basic_loading(self):
        """Test basic dataset loading functionality"""
        data = load_binary_with_probas(
            n_samples=self.n_samples, random_state=self.random_state
        )

        # Check return type and shape
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[0], self.n_samples)
        self.assertEqual(set(data.columns), self.expected_columns)

    def test_data_types_and_ranges(self):
        """Test data types and value ranges"""
        data = load_binary_with_probas(n_samples=self.n_samples)

        # Binary variables
        self.assertTrue(
            data[TREATMENT_COL].isin([0, 1]).all(), "Treatment should be binary"
        )
        self.assertTrue(
            data[OUTCOME_COL].isin([0, 1]).all(), "Outcome should be binary"
        )

        # Probability ranges
        prob_columns = [PS_COL, PROBAS_COL, PROBAS_T0_COL, PROBAS_T1_COL]
        for col in prob_columns:
            self.assertTrue(
                (data[col] >= 0).all() and (data[col] <= 1).all(),
                f"{col} should be between 0 and 1",
            )

    def test_covariate_distribution(self):
        """Test if covariates follow standard normal distribution"""
        data = load_binary_with_probas(n_samples=self.n_samples)

        for col in ["X1", "X2"]:
            self.assertAlmostEqual(data[col].mean(), 0, places=1)
            self.assertAlmostEqual(data[col].std(), 1, places=1)

    def test_return_params(self):
        """Test parameter return functionality"""
        data, params = load_binary_with_probas(
            n_samples=self.n_samples, return_params=True
        )

        # Check params structure
        self.assertIsInstance(params, dict)
        self.assertIn("treatment_params", params)
        self.assertIn("outcome_params", params)
        self.assertIn("DESCR", params)

        # Check parameter lengths
        self.assertEqual(len(params["treatment_params"]), 4)  # alpha parameters
        self.assertEqual(len(params["outcome_params"]), 5)  # beta parameters

    def test_reproducibility(self):
        """Test if random_state ensures reproducibility"""
        data1 = load_binary_with_probas(
            n_samples=self.n_samples, random_state=self.random_state
        )
        data2 = load_binary_with_probas(
            n_samples=self.n_samples, random_state=self.random_state
        )

        pd.testing.assert_frame_equal(data1, data2)

    def test_counterfactual_consistency(self):
        """Test if counterfactual probabilities are consistent"""
        data = load_binary_with_probas(
            n_samples=self.n_samples, random_state=self.random_state
        )

        # Y_prob should match Y_cf_1 when A=1
        np.testing.assert_array_almost_equal(
            data.loc[data[TREATMENT_COL] == 1, PROBAS_COL],
            data.loc[data[TREATMENT_COL] == 1, PROBAS_T1_COL],
        )

        # Y_prob should match Y_cf_0 when A=0
        np.testing.assert_array_almost_equal(
            data.loc[data[TREATMENT_COL] == 0, PROBAS_COL],
            data.loc[data[TREATMENT_COL] == 0, PROBAS_T0_COL],
        )

    def test_no_extreme_propensities(self):
        """Test if propensity scores are properly clipped"""
        data = load_binary_with_probas(n_samples=self.n_samples)

        self.assertTrue((data[PS_COL] >= 0.01).all(), "Minimum ps should be 0.01")
        self.assertTrue((data[PS_COL] <= 0.99).all(), "Maximum ps should be 0.99")

    def test_different_sample_sizes(self):
        """Test if different sample sizes work correctly"""
        test_sizes = [10, 100, 1000]
        for size in test_sizes:
            data = load_binary_with_probas(n_samples=size)
            self.assertEqual(
                data.shape[0], size, f"Expected {size} samples, got {data.shape[0]}"
            )


if __name__ == "__main__":
    unittest.main()
