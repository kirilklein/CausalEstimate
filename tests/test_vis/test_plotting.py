import unittest
import numpy as np
import pandas as pd

from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL, PROBAS_COL

try:
    import matplotlib.pyplot as plt
    from CausalEstimate.vis.plotting import (
        plot_hist_by_groups,
        plot_propensity_score_dist,
        plot_outcome_proba_dist,
    )

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
                PROBAS_COL: np.random.beta(2, 5, n),
            }
        )

    # -----------------------------
    # Existing tests for plot_propensity_score_dist
    # -----------------------------
    def test_plot_propensity_score_dist(self):
        fig, ax = plot_propensity_score_dist(self.df, PS_COL, TREATMENT_COL)

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "Propensity Score Distribution")
        self.assertEqual(ax.get_xlabel(), "Propensity Score")
        self.assertEqual(ax.get_ylabel(), "Count")
        self.assertEqual(len(ax.containers), 2)
        self.assertIsNotNone(ax.get_legend())

    def test_plot_propensity_score_dist_normalized(self):
        fig, ax = plot_propensity_score_dist(
            self.df, PS_COL, TREATMENT_COL, normalize=True
        )
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
        # The containers are the histogram bar groups
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

    # -----------------------------
    # NEW TESTS for _plot_hist_by_groups (optional)
    # -----------------------------
    def test_plot_hist_by_groups_basic(self):
        """
        Directly test the private _plot_hist_by_groups function
        for a basic scenario: 2 groups, default bins.
        """
        fig, ax = plot_hist_by_groups(
            df=self.df,
            value_col=PS_COL,
            group_col=TREATMENT_COL,
            group_values=(0, 1),
            group_labels=("Control", "Treatment"),
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        # We expect 2 hist plots
        self.assertEqual(len(ax.containers), 2)

    def test_plot_hist_by_groups_custom_bins(self):
        custom_bins = np.linspace(0, 1, 11)
        fig, ax = plot_hist_by_groups(
            df=self.df,
            value_col=PS_COL,
            group_col=TREATMENT_COL,
            bin_edges=custom_bins,
            group_values=(0, 1),
            group_labels=("Control", "Treatment"),
        )
        # Should be 10 bins
        self.assertEqual(len(ax.containers[0].patches), 10)

    def test_plot_hist_by_groups_existing_fig(self):
        fig, ax = plt.subplots()
        f2, a2 = plot_hist_by_groups(
            df=self.df, value_col=PS_COL, group_col=TREATMENT_COL, fig=fig, ax=ax
        )
        self.assertIs(fig, f2)
        self.assertIs(ax, a2)

    # -----------------------------
    # NEW TESTS for plot_outcome_proba_dist
    # -----------------------------
    def test_plot_outcome_proba_dist_basic(self):
        fig, ax = plot_outcome_proba_dist(self.df, PROBAS_COL, TREATMENT_COL)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "Outcome Probability Distribution")
        self.assertEqual(ax.get_xlabel(), "Predicted Outcome Probability")
        self.assertEqual(len(ax.containers), 2)

    def test_plot_outcome_proba_dist_normalized(self):
        fig, ax = plot_outcome_proba_dist(
            self.df, PROBAS_COL, TREATMENT_COL, normalize=True
        )
        self.assertEqual(ax.get_ylabel(), "Density")

    def test_plot_outcome_proba_dist_custom_bins(self):
        custom_bins = np.linspace(0, 1, 5)
        fig, ax = plot_outcome_proba_dist(
            self.df, PROBAS_COL, TREATMENT_COL, bin_edges=custom_bins
        )
        # We expect 4 bins
        self.assertEqual(len(ax.containers[0].patches), 4)

    def test_plot_outcome_proba_dist_custom_title_and_label(self):
        custom_title = "Custom Outcome Title"
        custom_xlabel = "Custom Outcome Label"
        fig, ax = plot_outcome_proba_dist(
            self.df,
            PROBAS_COL,
            TREATMENT_COL,
            title=custom_title,
            xlabel=custom_xlabel,
        )
        self.assertEqual(ax.get_title(), custom_title)
        self.assertEqual(ax.get_xlabel(), custom_xlabel)


if __name__ == "__main__":
    unittest.main()
