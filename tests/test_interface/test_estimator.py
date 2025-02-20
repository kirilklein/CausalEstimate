# test_multi_estimator.py

import unittest
import pandas as pd
import numpy as np

from CausalEstimate import MultiEstimator
from CausalEstimate.estimators.aipw import AIPW
from CausalEstimate.estimators.tmle import TMLE
from CausalEstimate.estimators.ipw import IPW
from CausalEstimate.utils.constants import (
    OUTCOME_COL,
    PS_COL,
    TREATMENT_COL,
    PROBAS_COL,
    PROBAS_T0_COL,
    PROBAS_T1_COL,
)


class TestMultiEstimator(unittest.TestCase):
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

        outcome_treated_probability = np.zeros_like(outcome_probability)
        outcome_treated_probability[treatment == 1] = outcome_probability[
            treatment == 1
        ]
        outcome_treated_probability[treatment == 0] = np.random.uniform(
            epsilon, 1 - epsilon, size
        )[treatment == 0]

        outcome_control_probability = np.zeros_like(outcome_probability)
        outcome_control_probability[treatment == 0] = outcome_probability[
            treatment == 0
        ]
        outcome_control_probability[treatment == 1] = np.random.uniform(
            epsilon, 1 - epsilon, size
        )[treatment == 1]

        cls.sample_data = pd.DataFrame(
            {
                TREATMENT_COL: treatment,
                OUTCOME_COL: outcome,
                PS_COL: propensity_score,
                PROBAS_COL: outcome_probability,
                PROBAS_T1_COL: outcome_treated_probability,
                PROBAS_T0_COL: outcome_control_probability,
            }
        )

    def _make_aipw(self):
        """
        Helper to instantiate an AIPW estimator for testing.
        """
        return AIPW(
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
            effect_type="ATE",
        )

    def _make_tmle(self):
        """
        Helper to instantiate a TMLE estimator for testing.
        """
        return TMLE(
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_col=PROBAS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
            effect_type="ATE",
        )

    def _make_ipw(self):
        """
        Helper to instantiate an IPW estimator for testing.
        """
        return IPW(
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            effect_type="ATE",
        )

    def test_compute_effect_no_bootstrap(self):
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        multi_estimator = MultiEstimator([aipw, tmle])

        # No bootstrap, no common support
        results = multi_estimator.compute_effects(
            df=self.sample_data,
            bootstrap=False,
            n_bootstraps=1,
            apply_common_support=False,
        )

        # Check that results are returned for both estimators
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)

        # Effect estimates should be floats
        self.assertIsInstance(results["AIPW"]["effect"], float)
        self.assertIsInstance(results["TMLE"]["effect"], float)

        # No std_err when bootstrap=False
        self.assertIsNone(results["AIPW"]["std_err"])
        self.assertIsNone(results["TMLE"]["std_err"])

        # Check that bootstrap flag is False
        self.assertFalse(results["AIPW"]["bootstrap"])
        self.assertFalse(results["TMLE"]["bootstrap"])

    def test_compute_effect_with_bootstrap(self):
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        multi_estimator = MultiEstimator([aipw, tmle])

        results = multi_estimator.compute_effects(
            df=self.sample_data,
            bootstrap=True,
            n_bootstraps=10,
            apply_common_support=False,
        )

        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)
        self.assertIsInstance(results["AIPW"]["effect"], float)
        self.assertIsInstance(results["TMLE"]["effect"], float)
        self.assertIsInstance(results["AIPW"]["std_err"], float)
        self.assertIsInstance(results["TMLE"]["std_err"], float)

        self.assertTrue(results["AIPW"]["bootstrap"])
        self.assertTrue(results["TMLE"]["bootstrap"])
        self.assertEqual(results["AIPW"]["n_bootstraps"], 10)
        self.assertEqual(results["TMLE"]["n_bootstraps"], 10)

    def test_missing_columns(self):
        # Here we just test that if columns are missing, the estimator complains.
        # For example, remove 'treatment' column:
        data_missing = self.sample_data.drop(columns=["treatment"])
        aipw = self._make_aipw()
        multi_estimator = MultiEstimator([aipw])

        with self.assertRaises(ValueError):
            multi_estimator.compute_effects(data_missing)

    def test_input_validation(self):
        # Introduce NaNs in the outcome column
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, "outcome"] = np.nan

        aipw = self._make_aipw()
        multi_estimator = MultiEstimator([aipw])

        with self.assertRaises(ValueError):
            multi_estimator.compute_effects(data_with_nan)

    def test_compute_effect_ipw(self):
        # IPW only needs treatment/outcome/ps
        ipw = self._make_ipw()
        multi_estimator = MultiEstimator([ipw])

        results = multi_estimator.compute_effects(self.sample_data)
        self.assertIn("IPW", results)
        self.assertIsInstance(results["IPW"]["effect"], float)

    def test_common_support_filtering(self):
        aipw = self._make_aipw()
        multi_estimator = MultiEstimator([aipw])

        results = multi_estimator.compute_effects(
            self.sample_data,
            apply_common_support=True,
            common_support_threshold=0.01,
            bootstrap=False,
            n_bootstraps=1,
        )
        self.assertIn("AIPW", results)
        self.assertIsInstance(results["AIPW"]["effect"], float)

    def test_parallel_bootstrap(self):
        # If you have parallel code internally, you can pass e.g. n_jobs=2 as part of your bootstrap routine.
        # We'll just test that normal bootstrapping works.
        aipw = self._make_aipw()
        multi_estimator = MultiEstimator([aipw])

        results = multi_estimator.compute_effects(
            df=self.sample_data,
            bootstrap=True,
            n_bootstraps=5,
        )
        self.assertTrue(results["AIPW"]["bootstrap"])
        self.assertEqual(results["AIPW"]["n_bootstraps"], 5)
        self.assertIsInstance(results["AIPW"]["std_err"], float)

    def test_multiple_estimators_including_ipw(self):
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        ipw = self._make_ipw()
        multi_estimator = MultiEstimator([aipw, tmle, ipw])

        results = multi_estimator.compute_effects(
            df=self.sample_data, bootstrap=False, n_bootstraps=1
        )
        # We expect a dict with all three
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)
        self.assertIn("IPW", results)


if __name__ == "__main__":
    unittest.main()
