import unittest
from CausalEstimate.simulation.binary_simulation import (
    simulate_binary_data,
    compute_ATE_theoretical_from_data,
    compute_ATE_theoretical_from_model,
)


class TestSimulation(unittest.TestCase):

    def test_simulation_data(self):
        alpha = [0.1, 0.2, -0.3, 0.5]
        beta = [0.5, 0.8, -0.6, 0.3, 0.2]
        data = simulate_binary_data(1000, alpha, beta, seed=42)

        self.assertEqual(data.shape[0], 1000)
        self.assertTrue("A" in data.columns and "Y" in data.columns)

    def test_ATE_theoretical(self):
        alpha = [0.1, 0.2, -0.3, 0.5]
        beta = [0.5, 0.8, -0.6, 0.3, 0.2]
        data = simulate_binary_data(1000, alpha, beta, seed=42)
        ate_data = compute_ATE_theoretical_from_data(data, beta)
        ate_model = compute_ATE_theoretical_from_model(beta)
        self.assertAlmostEqual(
            ate_data, ate_model, delta=0.05
        )  # Example expected ATE value


if __name__ == "__main__":
    unittest.main()
