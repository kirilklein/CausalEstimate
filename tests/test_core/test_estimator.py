import unittest
import pandas as pd
import numpy as np

from CausalEstimate.core.multi_estimator import MultiEstimator
from CausalEstimate.estimators.aipw import AIPW
from CausalEstimate.estimators.tmle import TMLE
from CausalEstimate.estimators.ipw import IPW
from CausalEstimate.utils.constants import (
    OUTCOME_COL,
    PS_COL,
    TREATMENT_COL,
    PROBAS_COL,
    PROBAS_T1_COL,
    PROBAS_T0_COL,
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
)


class TestMultiEstimatorCombined(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate sample data for testing
        np.random.seed(42)
        size = 500
        epsilon = 1e-2  # Small value to avoid 0/1 extremes
        propensity_score = np.random.uniform(epsilon, 1 - epsilon, size)
        outcome_probability = np.random.uniform(epsilon, 1 - epsilon, size)
        treatment = np.random.binomial(1, propensity_score, size)
        outcome = np.random.binomial(1, outcome_probability, size)

        # For treated and untreated probabilities, create simple placeholders
        outcome_treated_probability = np.where(
            treatment == 1,
            outcome_probability,
            np.random.uniform(epsilon, 1 - epsilon, size),
        )
        outcome_control_probability = np.where(
            treatment == 0,
            outcome_probability,
            np.random.uniform(epsilon, 1 - epsilon, size),
        )

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
        Creates a TMLE estimator instance configured to estimate the Average Treatment effect on the Treated (ATT) using predefined column names.
        """
        return TMLE(
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            probas_col=PROBAS_COL,
            probas_t1_col=PROBAS_T1_COL,
            probas_t0_col=PROBAS_T0_COL,
            effect_type="ATT",
        )

    def _make_ipw(self):
        """
        Creates and returns an IPW estimator configured for Average Treatment Effect (ATE).

        The estimator uses predefined column names for treatment, outcome, and propensity scores.
        """
        return IPW(
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            effect_type="ATE",
        )

    def _assert_result_structure(
        self, result, bootstrap_requested=False, n_bootstraps=1
    ):
        """
        Helper to assert that the result dictionary has the expected structure.
        When bootstrap is not applied (n_bootstraps=1), we expect a minimal summary.
        """
        # Basic keys we expect in all cases (at least the effect)
        self.assertIn(EFFECT, result)
        # When bootstrapping is not applied, we expect n_bootstraps to be 0.
        if n_bootstraps == 1:
            self.assertEqual(result["n_bootstraps"], 0)
        else:
            self.assertEqual(result["n_bootstraps"], n_bootstraps)
            # If bootstrap samples were requested, ensure they are present;
            # otherwise, they should be absent.
            if bootstrap_requested:
                self.assertIn("bootstrap_samples", result)
                bs = result["bootstrap_samples"]
                for key in [EFFECT, EFFECT_treated, EFFECT_untreated]:
                    self.assertIn(key, bs)
                    self.assertEqual(len(bs[key]), n_bootstraps)
            else:
                self.assertNotIn("bootstrap_samples", result)

    def test_compute_effect_no_bootstrap(self):
        """Test that when n_bootstraps=1 (no bootstrap), results have the expected structure."""
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        multi_est = MultiEstimator([aipw, tmle])

        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=1,
            apply_common_support=False,
        )
        # Check that results exist for both estimators
        for estimator_key in ["AIPW", "TMLE"]:
            with self.subTest(estimator=estimator_key):
                self.assertIn(estimator_key, results)
                self._assert_result_structure(
                    results[estimator_key], bootstrap_requested=False, n_bootstraps=1
                )

    def test_compute_effect_with_bootstrap(self):
        """Test that when n_bootstraps > 1 and bootstrap samples are not requested, the summary is computed correctly."""
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        multi_est = MultiEstimator([aipw, tmle])

        n_boot = 10
        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=n_boot,
            apply_common_support=False,
            return_bootstrap_samples=False,
        )
        for estimator_key in ["AIPW", "TMLE"]:
            with self.subTest(estimator=estimator_key):
                self._assert_result_structure(
                    results[estimator_key],
                    bootstrap_requested=False,
                    n_bootstraps=n_boot,
                )

    def test_bootstrap_with_samples_flag(self):
        """Test that bootstrap samples are included when requested."""
        tmle = self._make_tmle()
        multi_est = MultiEstimator([tmle])
        n_boot = 10

        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=n_boot,
            apply_common_support=False,
            return_bootstrap_samples=True,
        )
        res = results["TMLE"]
        self._assert_result_structure(
            res, bootstrap_requested=True, n_bootstraps=n_boot
        )

    def test_missing_columns(self):
        """Test that a missing required column (e.g., treatment column) raises an error."""
        data_missing = self.sample_data.drop(columns=[TREATMENT_COL])
        aipw = self._make_aipw()
        multi_est = MultiEstimator([aipw])
        with self.assertRaises(ValueError):
            multi_est.compute_effects(data_missing)

    def test_input_validation(self):
        """Test that input data with NaNs (e.g., in the outcome column) triggers an error."""
        data_nan = self.sample_data.copy()
        data_nan.loc[0, OUTCOME_COL] = np.nan
        aipw = self._make_aipw()
        multi_est = MultiEstimator([aipw])
        with self.assertRaises(ValueError):
            multi_est.compute_effects(data_nan)

    def test_common_support_filtering(self):
        """Test that enabling common support filtering still returns valid effect estimates."""
        aipw = self._make_aipw()
        multi_est = MultiEstimator([aipw])
        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=1,
            apply_common_support=True,
            common_support_threshold=0.01,
        )
        res = results["AIPW"]
        self.assertIn(EFFECT, res)
        self.assertIsInstance(res[EFFECT], float)

    def test_compute_effect_ipw(self):
        """Test that an IPW estimator (with minimal required columns) returns a valid effect."""
        ipw = self._make_ipw()
        multi_est = MultiEstimator([ipw])
        results = multi_est.compute_effects(self.sample_data)
        self.assertIn("IPW", results)
        self.assertIsInstance(results["IPW"][EFFECT], float)

    def test_multiple_estimators_including_ipw(self):
        """Test that when multiple estimators are provided, all keys are returned."""
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        ipw = self._make_ipw()
        multi_est = MultiEstimator([aipw, tmle, ipw])
        results = multi_est.compute_effects(df=self.sample_data, n_bootstraps=1)
        for estimator_name in ["AIPW", "TMLE", "IPW"]:
            with self.subTest(estimator=estimator_name):
                self.assertIn(estimator_name, results)


if __name__ == "__main__":
    unittest.main()
