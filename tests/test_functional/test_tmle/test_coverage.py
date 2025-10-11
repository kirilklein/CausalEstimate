import unittest

import numpy as np

from CausalEstimate.estimators.functional.tmle import (
    compute_tmle_ate,
    compute_tmle_rr,
)
from CausalEstimate.estimators.functional.tmle_att import (
    compute_tmle_att,
)
from CausalEstimate.utils.constants import CI95_LOWER, CI95_UPPER
from tests.helpers.setup import generate_simulation_data


class TestTMLECoverage(unittest.TestCase):
    """
    Tests CI coverage when models are misspecified.
    This validates double robustness.
    """

    n_simulations = 300
    n_samples = 2000
    alpha = [-0.2, 0.5, -0.5]
    beta = [0.1, 0.4, 0.6, -2]
    noise_level = 0.0

    def test_ate_coverage(self):
        """Test if the ATE 95% CI covers the true value ~95% of the time."""
        coverage_count = 0

        for i in range(self.n_simulations):
            sim_data = generate_simulation_data(
                n=self.n_samples,
                alpha=self.alpha,
                beta=self.beta,
                noise_level=self.noise_level,
                seed=i,  # Use loop index as the seed for reproducibility
            )

            result = compute_tmle_ate(
                sim_data["A"],
                sim_data["Y"],
                sim_data["ps"],
                sim_data["Y0_hat"],
                sim_data["Y1_hat"],
                sim_data["Yhat"],
            )

            true_ate = sim_data["true_ate"]
            if result[CI95_LOWER] is not None and np.isfinite(result[CI95_LOWER]):
                if result[CI95_LOWER] <= true_ate <= result[CI95_UPPER]:
                    coverage_count += 1

        coverage_probability = coverage_count / self.n_simulations
        print(f"\nATE Coverage: {coverage_probability:.3f}")

        self.assertGreaterEqual(coverage_probability, 0.94)

    def test_att_coverage(self):
        """Test ATT coverage with misspecified nuisance models."""
        coverage_count = 0
        for i in range(self.n_simulations):
            sim_data = generate_simulation_data(
                n=self.n_samples,
                alpha=self.alpha,
                beta=self.beta,
                noise_level=self.noise_level,
                seed=i,
            )
            result = compute_tmle_att(
                sim_data["A"],
                sim_data["Y"],
                sim_data["ps"],
                sim_data["Y0_hat"],
                sim_data["Y1_hat"],
                sim_data["Yhat"],
            )
            true_att = sim_data["true_att"]
            if np.isfinite(result[CI95_LOWER]) and (
                result[CI95_LOWER] <= true_att <= result[CI95_UPPER]
            ):
                coverage_count += 1

        coverage_probability = coverage_count / self.n_simulations
        print(f"ATT Coverage: {coverage_probability:.3f}")
        self.assertGreaterEqual(coverage_probability, 0.94)

    def test_rr_coverage(self):
        """Test RR coverage with misspecified nuisance models."""
        coverage_count = 0
        for i in range(self.n_simulations):
            sim_data = generate_simulation_data(
                n=self.n_samples,
                alpha=self.alpha,
                beta=self.beta,
                noise_level=self.noise_level,
                seed=i,
            )
            result = compute_tmle_rr(
                sim_data["A"],
                sim_data["Y"],
                sim_data["ps"],
                sim_data["Y0_hat"],
                sim_data["Y1_hat"],
                sim_data["Yhat"],
            )
            true_rr = sim_data["true_rr"]
            if np.isfinite(result[CI95_LOWER]) and (
                result[CI95_LOWER] <= true_rr <= result[CI95_UPPER]
            ):
                coverage_count += 1

        coverage_probability = coverage_count / self.n_simulations
        print(f"RR Coverage: {coverage_probability:.3f}")
        self.assertGreaterEqual(coverage_probability, 0.94)


class TestTMLECoverageMisspecified(unittest.TestCase):
    """
    Tests CI coverage when models are misspecified.
    This validates double robustness.
    """

    n_simulations = 200
    n_samples = 2000
    alpha = [-0.2, 0.5, -0.5, 2]
    beta = [0.1, 0.4, 0.6, -2, 0.1]
    noise_level = 0.01

    def test_ate_coverage(self):
        """Test if the ATE 95% CI covers the true value ~95% of the time."""
        coverage_count = 0

        for i in range(self.n_simulations):
            sim_data = generate_simulation_data(
                n=self.n_samples,
                alpha=self.alpha,
                beta=self.beta,
                noise_level=self.noise_level,
                seed=i,  # Use loop index as the seed for reproducibility
            )

            result = compute_tmle_ate(
                sim_data["A"],
                sim_data["Y"],
                sim_data["ps"],
                sim_data["Y0_hat"],
                sim_data["Y1_hat"],
                sim_data["Yhat"],
            )

            true_ate = sim_data["true_ate"]
            if result[CI95_LOWER] is not None and np.isfinite(result[CI95_LOWER]):
                if result[CI95_LOWER] <= true_ate <= result[CI95_UPPER]:
                    coverage_count += 1

        coverage_probability = coverage_count / self.n_simulations
        print(f"\nATE Coverage (Misspecified): {coverage_probability:.3f}")

        self.assertGreaterEqual(coverage_probability, 0.92)

    def test_att_coverage(self):
        """Test ATT coverage with misspecified nuisance models."""
        coverage_count = 0
        for i in range(self.n_simulations):
            sim_data = generate_simulation_data(
                n=self.n_samples,
                alpha=self.alpha,
                beta=self.beta,
                noise_level=self.noise_level,
                seed=i,
            )
            result = compute_tmle_att(
                sim_data["A"],
                sim_data["Y"],
                sim_data["ps"],
                sim_data["Y0_hat"],
                sim_data["Y1_hat"],
                sim_data["Yhat"],
            )
            true_att = sim_data["true_att"]
            if np.isfinite(result[CI95_LOWER]) and (
                result[CI95_LOWER] <= true_att <= result[CI95_UPPER]
            ):
                coverage_count += 1

        coverage_probability = coverage_count / self.n_simulations
        print(f"ATT Coverage (Misspecified): {coverage_probability:.3f}")
        self.assertGreaterEqual(coverage_probability, 0.90)

    def test_rr_coverage(self):
        """Test RR coverage with misspecified nuisance models."""
        coverage_count = 0
        for i in range(self.n_simulations):
            sim_data = generate_simulation_data(
                n=self.n_samples,
                alpha=self.alpha,
                beta=self.beta,
                noise_level=self.noise_level,
                seed=i,
            )
            result = compute_tmle_rr(
                sim_data["A"],
                sim_data["Y"],
                sim_data["ps"],
                sim_data["Y0_hat"],
                sim_data["Y1_hat"],
                sim_data["Yhat"],
            )
            true_rr = sim_data["true_rr"]
            if np.isfinite(result[CI95_LOWER]) and (
                result[CI95_LOWER] <= true_rr <= result[CI95_UPPER]
            ):
                coverage_count += 1

        coverage_probability = coverage_count / self.n_simulations
        print(f"RR Coverage (Misspecified): {coverage_probability:.3f}")
        self.assertGreaterEqual(coverage_probability, 0.90)


if __name__ == "__main__":
    unittest.main()
