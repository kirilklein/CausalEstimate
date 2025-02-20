import unittest

from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from CausalEstimate.utils.constants import OUTCOME_COL, TREATMENT_COL


class TestSimulation(unittest.TestCase):

    def test_simulation_data(self):
        alpha = [0.1, 0.2, -0.3, 0.5]
        beta = [0.5, 0.8, -0.6, 0.3, 0.2]
        data = simulate_binary_data(100, alpha, beta, seed=42)

        self.assertEqual(data.shape[0], 100)
        self.assertTrue(TREATMENT_COL in data.columns and OUTCOME_COL in data.columns)


if __name__ == "__main__":
    unittest.main()
