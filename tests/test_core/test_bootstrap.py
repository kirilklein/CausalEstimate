import unittest
import pandas as pd
import numpy as np
from CausalEstimate.core.bootstrap import generate_bootstrap_samples


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        # Create a simple test DataFrame
        self.test_df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
        self.n_bootstraps = 3

    def test_generate_bootstrap_samples_output_structure(self):
        # Test the structure of the output
        bootstrap_samples = generate_bootstrap_samples(self.test_df, self.n_bootstraps)

        # Check if the output is a list
        self.assertIsInstance(bootstrap_samples, list)

        # Check if the number of bootstrap samples is correct
        self.assertEqual(len(bootstrap_samples), self.n_bootstraps)

        # Check if each bootstrap sample is a DataFrame with the same shape
        for sample in bootstrap_samples:
            self.assertIsInstance(sample, pd.DataFrame)
            self.assertEqual(sample.shape, self.test_df.shape)
            self.assertEqual(list(sample.columns), list(self.test_df.columns))

    def test_generate_bootstrap_samples_with_seed(self):
        # Test reproducibility with a fixed seed
        np.random.seed(42)
        samples1 = generate_bootstrap_samples(self.test_df, 1)[0]

        np.random.seed(42)
        samples2 = generate_bootstrap_samples(self.test_df, 1)[0]

        # Check if the samples are identical when using the same seed
        pd.testing.assert_frame_equal(samples1, samples2)

    def test_generate_bootstrap_samples_different_samples(self):
        # Test that bootstrap samples are actually different
        samples = generate_bootstrap_samples(self.test_df, 2)

        # Check that the two samples are different (they should be, with very high probability)
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(samples[0], samples[1])

    def test_generate_bootstrap_samples_with_empty_df(self):
        # Test behavior with empty DataFrame
        empty_df = pd.DataFrame(columns=["A", "B"])
        samples = generate_bootstrap_samples(empty_df, self.n_bootstraps)

        self.assertEqual(len(samples), self.n_bootstraps)
        for sample in samples:
            self.assertEqual(len(sample), 0)


if __name__ == "__main__":
    unittest.main()
