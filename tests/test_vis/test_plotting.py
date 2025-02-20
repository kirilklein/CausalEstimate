import unittest

import numpy as np
import pandas as pd

from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL

try:
    import matplotlib.pyplot as plt

    from CausalEstimate.vis.plotting import plot_propensity_score_dist

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib is not installed")
class TestPlotting(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)
        n = 1000
        self.df = pd.DataFrame(
            {
                PS_COL: np.random.beta(2, 5, n),
                TREATMENT_COL: np.random.binomial(1, 0.3, n),
            }
        )

    def test_plot_propensity_score_dist(self):
        fig, ax = plot_propensity_score_dist(self.df, PS_COL, TREATMENT_COL)

        # Check that the figure and axis objects are created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        # Check that the title and labels are set correctly
        self.assertEqual(ax.get_title(), "Propensity Score Distribution")
        self.assertEqual(ax.get_xlabel(), "Propensity Score")
        self.assertEqual(ax.get_ylabel(), "Count")

        # Check that two histograms are plotted (one for each group)
        self.assertEqual(len(ax.containers), 2)

        # Check that the legend is present
        self.assertIsNotNone(ax.get_legend())

    def test_plot_propensity_score_dist_normalized(self):
        fig, ax = plot_propensity_score_dist(
            self.df, PS_COL, TREATMENT_COL, normalize=True
        )

        # Check that the y-label is changed to "Density" when normalized
        self.assertEqual(ax.get_ylabel(), "Density")

    def test_plot_propensity_score_dist_custom_params(self):
        custom_title = "Custom Title"
        custom_xlabel = "Custom X Label"
        custom_bins = np.linspace(0, 1, 21)  # 20 bins

        fig, ax = plot_propensity_score_dist(
            self.df,
            PS_COL,
            TREATMENT_COL,
            title=custom_title,
            xlabel=custom_xlabel,
            bin_edges=custom_bins,
        )

        self.assertEqual(ax.get_title(), custom_title)
        self.assertEqual(ax.get_xlabel(), custom_xlabel)

        # Check that the number of bins is correct
        self.assertEqual(len(ax.containers[0].patches), 20)

    def test_plot_propensity_score_dist_existing_fig_ax(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_propensity_score_dist(
            self.df, PS_COL, TREATMENT_COL, fig=fig, ax=ax
        )

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)

    def test_plot_propensity_score_dist_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_propensity_score_dist(
                self.df, PS_COL, TREATMENT_COL, fig=None, ax=plt.gca()
            )


if __name__ == "__main__":
    unittest.main()
