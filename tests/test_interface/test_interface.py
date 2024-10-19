# test_estimator.py

import unittest
import pandas as pd
import numpy as np
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.estimators.aipw import AIPW
from CausalEstimate.estimators.tmle import TMLE


class TestEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate sample data once for all tests
        np.random.seed(42)
        size = 500
        epsilon = 1e-3  # Small value to avoid exact 0 or 1
        propensity_score = np.random.uniform(epsilon, 1 - epsilon, size)
        outcome_probability = np.random.uniform(epsilon, 1 - epsilon, size)
        treatment = np.random.binomial(1, propensity_score, size)
        outcome = np.random.binomial(1, outcome_probability, size)
        outcome_treated_probability = np.zeros_like(outcome)
        outcome_treated_probability[treatment == 1] = outcome_probability[
            treatment == 1
        ]
        outcome_treated_probability[treatment == 0] = np.random.uniform(
            epsilon, 1 - epsilon, size
        )[treatment == 0]
        outcome_control_probability = np.zeros_like(outcome)
        outcome_control_probability[treatment == 0] = outcome_probability[
            treatment == 0
        ]
        outcome_control_probability[treatment == 1] = np.random.uniform(
            epsilon, 1 - epsilon, size
        )[treatment == 1]

        cls.sample_data = pd.DataFrame(
            {
                "treatment": treatment,
                "outcome": outcome,
                "propensity_score": propensity_score,
                "predicted_outcome": outcome_probability,
                "predicted_outcome_treated": outcome_treated_probability,
                "predicted_outcome_control": outcome_control_probability,
            }
        )
        cls.method_args = {
            "AIPW": {
                "predicted_outcome_treated_col": "predicted_outcome_treated",
                "predicted_outcome_control_col": "predicted_outcome_control",
            },
            "TMLE": {
                "predicted_outcome_treated_col": "predicted_outcome_treated",
                "predicted_outcome_control_col": "predicted_outcome_control",
                "predicted_outcome_col": "predicted_outcome",
            },
        }

    def test_compute_effect_no_bootstrap(self):
        estimator = Estimator(methods=["AIPW", "TMLE"], effect_type="ATE")
        # Define estimator-specific arguments
        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            method_args=self.method_args,
        )

        # Check that results are returned for all specified methods
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)

        # Check that the effect estimates are floats
        self.assertIsInstance(results["AIPW"]["effect"], float)
        self.assertIsInstance(results["TMLE"]["effect"], float)

        # Check that standard errors are None when bootstrap=False
        self.assertIsNone(results["AIPW"]["std_err"])
        self.assertIsNone(results["TMLE"]["std_err"])

        # Check that bootstrap flag is False
        self.assertFalse(results["AIPW"]["bootstrap"])
        self.assertFalse(results["TMLE"]["bootstrap"])

    def test_compute_effect_with_bootstrap(self):
        estimator = Estimator(methods=["AIPW", "TMLE"], effect_type="ATE")
        # Define estimator-specific arguments
        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            bootstrap=True,
            n_bootstraps=10,
            method_args=self.method_args,
        )

        # Check that results are returned for all specified methods
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)

        # Check that the effect estimates are floats
        self.assertIsInstance(results["AIPW"]["effect"], float)
        self.assertIsInstance(results["TMLE"]["effect"], float)

        # Check that standard errors are floats
        self.assertIsInstance(results["AIPW"]["std_err"], float)
        self.assertIsInstance(results["TMLE"]["std_err"], float)

        # Check that bootstrap flag is True
        self.assertTrue(results["AIPW"]["bootstrap"])
        self.assertTrue(results["TMLE"]["bootstrap"])

        # Check that the number of bootstraps is correct
        self.assertEqual(results["AIPW"]["n_bootstraps"], 10)
        self.assertEqual(results["TMLE"]["n_bootstraps"], 10)

    def test_estimator_specific_params(self):
        method_params = {
            "AIPW": {"some_param": 1},
            "TMLE": {"another_param": 2},
        }
        estimator = Estimator(
            methods=["AIPW", "TMLE"], effect_type="ATE", method_params=method_params
        )
        # Define estimator-specific arguments

        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            method_args=self.method_args,
        )

        # Ensure the code runs without errors
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)

    def test_missing_columns(self):
        estimator = Estimator(methods=["AIPW"], effect_type="ATE")
        # Remove the 'treatment' column to simulate missing data
        sample_data_missing = self.sample_data.drop(columns=["treatment"])
        # Define estimator-specific arguments
        with self.assertRaises(ValueError) as context:
            estimator.compute_effect(
                sample_data_missing,
                treatment_col="treatment",
                outcome_col="outcome",
                ps_col="propensity_score",
                bootstrap=False,
                method_args=self.method_args,
            )
        self.assertTrue(context.exception)

    def test_invalid_method(self):
        with self.assertRaises(ValueError) as context:
            Estimator(methods=["InvalidMethod"], effect_type="ATE")
        self.assertIn(
            "Method 'InvalidMethod' is not supported.", str(context.exception)
        )

    def test_estimator_access(self):
        estimator = Estimator(methods=["AIPW", "TMLE"], effect_type="ATE")
        # Define estimator-specific arguments

        estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            method_args=self.method_args,
        )

        # Access the AIPW estimator instance
        aipw_estimator = next(e for e in estimator.estimators if isinstance(e, AIPW))
        self.assertIsInstance(aipw_estimator, AIPW)

        # Access the TMLE estimator instance
        tmle_estimator = next(e for e in estimator.estimators if isinstance(e, TMLE))
        self.assertIsInstance(tmle_estimator, TMLE)

    def test_parallel_bootstrapping(self):
        estimator = Estimator(methods=["AIPW"], effect_type="ATE")
        # Define estimator-specific arguments

        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            bootstrap=True,
            n_bootstraps=10,
            method_args=self.method_args,
            # Include n_jobs parameter if your implementation supports parallel processing
        )

        # Check that bootstrapping results are returned
        self.assertIn("AIPW", results)
        self.assertTrue(results["AIPW"]["bootstrap"])
        self.assertEqual(results["AIPW"]["n_bootstraps"], 10)
        self.assertIsInstance(results["AIPW"]["std_err"], float)

    def test_input_validation(self):
        estimator = Estimator(methods=["AIPW"], effect_type="ATE")
        # Introduce NaN values in the outcome column
        sample_data_with_nan = self.sample_data.copy()
        sample_data_with_nan.loc[0, "outcome"] = np.nan
        # Define estimator-specific arguments

        with self.assertRaises(ValueError) as context:
            estimator.compute_effect(
                sample_data_with_nan,
                treatment_col="treatment",
                outcome_col="outcome",
                ps_col="propensity_score",
                method_args=self.method_args,
            )
        self.assertIsInstance(context.exception, ValueError)

    def test_compute_effect_with_additional_columns(self):
        # Assuming IPW requires 'propensity_score' column
        estimator = Estimator(methods=["IPW"], effect_type="ATE")
        # Define estimator-specific arguments

        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
        )
        self.assertIn("IPW", results)
        self.assertIsInstance(results["IPW"]["effect"], float)

    def test_compute_effect_without_method_args(self):
        estimator = Estimator(methods=["IPW"], effect_type="ATE")
        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
        )
        self.assertIn("IPW", results)
        self.assertIsInstance(results["IPW"]["effect"], float)

    def test_common_support_filtering(self):
        estimator = Estimator(methods=["AIPW"], effect_type="ATE")
        # Define estimator-specific arguments

        results = estimator.compute_effect(
            self.sample_data,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            method_args=self.method_args,
            apply_common_support=True,
            common_support_threshold=0.01,
        )
        self.assertIn("AIPW", results)
        self.assertIsInstance(results["AIPW"]["effect"], float)

    def test_matching(self):
        df = self.sample_data.copy()
        df["treatment"] = np.random.binomial(1, 0.1, size=len(df))
        estimator = Estimator(methods=["MATCHING"], effect_type="ATE")
        results = estimator.compute_effect(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
        )
        self.assertIn("MATCHING", results)

    def test_matching_bootstrap(self):
        df = self.sample_data.copy()
        df["treatment"] = np.random.binomial(1, 0.1, size=len(df))
        estimator = Estimator(methods=["MATCHING"], effect_type="ATE")
        results = estimator.compute_effect(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            bootstrap=True,
            n_bootstraps=10,
            method_args={"MATCHING": {"caliper": 0.05}},
        )
        self.assertIn("MATCHING", results)

    def test_combined_bootstrap(self):
        df = self.sample_data.copy()
        df["treatment"] = np.random.binomial(1, 0.1, size=len(df))
        estimator = Estimator(
            methods=["AIPW", "MATCHING", "IPW", "TMLE"], effect_type="ATE"
        )

        results = estimator.compute_effect(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            ps_col="propensity_score",
            bootstrap=True,
            n_bootstraps=10,
            method_args=self.method_args,
        )
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)
        self.assertIn("MATCHING", results)
        self.assertIn("IPW", results)


if __name__ == "__main__":
    unittest.main()
