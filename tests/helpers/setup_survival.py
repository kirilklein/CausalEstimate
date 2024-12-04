import unittest

import numpy as np

from CausalEstimate.simulation.survival_simulation import simulate_data, theoretical_survival_function
from helpers.models import fit_ps_model, fit_failure_model, fit_censoring_model, estimate_survival_probability_at_T
from helpers.survival_utils import transform_data_for_model_estimation


# Base test class for survival estimators
class TestSurvivalEffectBase(unittest.TestCase):
    n: int = 2000
    seed: int = 42

    @classmethod
    def setUpClass(cls):
        # Simulate data
        cls.data = simulate_data(cls.n)
        cls.data["pid"] = np.arange(len(cls.data))
        cls.A = cls.data["A"].values
        cls.X = cls.data["X"].values
        cls.T_observed = cls.data["T_observed"].values
        cls.Y = cls.data["Y"].values

        # Compute theoretical survival function
        max_time = int(cls.T_observed.max())
        cls.t_values = np.arange(1, max_time + 1)
        cls.theoretical_surv = theoretical_survival_function(
            t=cls.t_values, X=cls.X, A=cls.A
        )
        cls.average_theoretical_surv = np.mean(cls.theoretical_surv, axis=0)

        # Estimate propensity scores
        cls.data = fit_ps_model(cls.data)

        # Transform data for model estimation
        cls.cls_data = transform_data_for_model_estimation(cls.data)

        # Fit failure and censoring models
        cls.failure_model = fit_failure_model(cls.cls_data)
        cls.censoring_model = fit_censoring_model(cls.cls_data)

        # Estimate survival probabilities
        cls.data["failure_survival_prob"] = estimate_survival_probability_at_T(
            cls.data, cls.failure_model
        )
        cls.data["censoring_survival_prob"] = estimate_survival_probability_at_T(
            cls.data, cls.censoring_model
        )

        # Prepare data for IPCW estimator
        ps_col = "ps"
        survival_prob_col = "failure_survival_prob"
        time_col = "T_observed"

        cls.ps_col = ps_col
        cls.survival_prob_col = survival_prob_col
        cls.time_col = time_col


# Example usage
if __name__ == "__main__":
    unittest.main()