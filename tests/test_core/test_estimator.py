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
        epsilon = 1e-3  # Small value to avoid 0/1 extremes
        propensity_score = np.random.uniform(epsilon, 1 - epsilon, size)
        outcome_probability = np.random.uniform(epsilon, 1 - epsilon, size)
        treatment = np.random.binomial(1, propensity_score, size)
        outcome = np.random.binomial(1, outcome_probability, size)

        # For treated and untreated probabilities, we create simple placeholders
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
        return IPW(
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            effect_type="ATE",
        )

    def test_compute_effect_no_bootstrap(self):
        """Test that when n_bootstraps=1 (no bootstrap), results have the expected keys and flags."""
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        multi_est = MultiEstimator([aipw, tmle])

        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=1,
            apply_common_support=False,
        )
        # Check that results exist for both estimators
        self.assertIn("AIPW", results)
        self.assertIn("TMLE", results)

        for estimator_key in ["AIPW", "TMLE"]:
            res = results[estimator_key]
            # Expect summary keys to be present
            for key in [EFFECT]:
                self.assertIn(key, res)
            # With n_bootstraps=1, bootstrapping was not applied
            self.assertEqual(res["n_bootstraps"], 0)

    def test_compute_effect_with_bootstrap(self):
        """Test that when n_bootstraps>1 and no bootstrap samples flag, the summary is computed correctly."""
        aipw = self._make_aipw()
        tmle = self._make_tmle()
        multi_est = MultiEstimator([aipw, tmle])

        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=10,
            apply_common_support=False,
            return_bootstrap_samples=False,
        )
        for estimator_key in ["AIPW", "TMLE"]:
            res = results[estimator_key]
            self.assertEqual(res["n_bootstraps"], 10)
            # When bootstrap samples are not requested, the key should not be present
            self.assertNotIn("bootstrap_samples", res)

    def test_bootstrap_with_samples_flag(self):
        """Test that the bootstrap samples are included when requested."""
        tmle = self._make_tmle()
        multi_est = MultiEstimator([tmle])

        results = multi_est.compute_effects(
            df=self.sample_data,
            n_bootstraps=10,
            apply_common_support=False,
            return_bootstrap_samples=True,
        )
        res = results["TMLE"]
        self.assertEqual(res["n_bootstraps"], 10)
        self.assertIn("bootstrap_samples", res)
        bs_samples = res["bootstrap_samples"]
        for key in [EFFECT, EFFECT_treated, EFFECT_untreated]:
            self.assertIn(key, bs_samples)
            self.assertEqual(len(bs_samples[key]), 10)

    def test_missing_columns(self):
        """Test that a missing required column raises an error (e.g. missing treatment column)."""
        data_missing = self.sample_data.drop(columns=[TREATMENT_COL])
        aipw = self._make_aipw()
        multi_est = MultiEstimator([aipw])
        with self.assertRaises(ValueError):
            multi_est.compute_effects(data_missing)

    def test_input_validation(self):
        """Test that input data with NaNs (e.g. in the outcome column) triggers an error."""
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
            self.assertIn(estimator_name, results)


if __name__ == "__main__":
    unittest.main()
