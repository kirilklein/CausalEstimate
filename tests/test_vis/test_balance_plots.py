import unittest

import numpy as np
import pandas as pd

from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from CausalEstimate.diagnostics import compute_balance_table
    from CausalEstimate.vis.plotting import (
        plot_love,
        plot_ps_boxplot,
        plot_weight_dist,
        plot_zipper,
    )

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib is not installed")
class TestBalancePlots(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 500
        ps = np.clip(rng.beta(2, 5, n), 1e-6, 1 - 1e-6)
        self.df = pd.DataFrame(
            {
                PS_COL: ps,
                TREATMENT_COL: rng.binomial(1, ps),
                "X1": rng.normal(size=n),
                "X2": rng.normal(size=n),
            }
        )

    def tearDown(self):
        plt.close("all")

    def test_plot_weight_dist_returns_fig_ax(self):
        fig, ax = plot_weight_dist(self.df)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_weight_dist_reuses_axes(self):
        fig, ax = plt.subplots()
        fig2, ax2 = plot_weight_dist(self.df, fig=fig, ax=ax)
        self.assertIs(ax2, ax)

    def test_plot_weight_dist_degenerate_ps_raises(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], PS_COL] = 0.0
        with self.assertRaises(ValueError):
            plot_weight_dist(bad)

    def test_plot_love_ticks_match_covariates(self):
        table = compute_balance_table(self.df, ["X1", "X2"])
        fig, ax = plot_love(table)
        self.assertEqual(len(ax.get_yticks()), 2)
        labels = {t.get_text() for t in ax.get_yticklabels()}
        self.assertEqual(labels, {"X1", "X2"})

    def test_plot_weight_dist_constant_weights_normalized(self):
        df = pd.DataFrame({TREATMENT_COL: [1, 1, 0, 0] * 5, PS_COL: [0.5] * 20})
        fig, ax = plot_weight_dist(df, normalize=True)
        heights = [p.get_height() for p in ax.patches]
        self.assertTrue(np.all(np.isfinite(heights)))

    def test_plot_love_all_nan_raises(self):
        table = pd.DataFrame(
            {"smd_unweighted": [np.nan], "smd_weighted": [np.nan]}, index=["const"]
        )
        with self.assertWarns(RuntimeWarning):
            with self.assertRaises(ValueError):
                plot_love(table)

    def test_plot_love_drops_nan_smd_with_warning(self):
        df = self.df.copy()
        df["const"] = 1.0
        with self.assertWarns(RuntimeWarning):
            table = compute_balance_table(df, ["X1", "const"])
        with self.assertWarns(RuntimeWarning):
            fig, ax = plot_love(table)
        self.assertEqual(len(ax.get_yticks()), 1)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib is not installed")
class TestPsBoxplot(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        ps = np.clip(rng.beta(2, 5, n), 1e-6, 1 - 1e-6)
        self.df = pd.DataFrame({PS_COL: ps, TREATMENT_COL: rng.binomial(1, ps)})

    def tearDown(self):
        plt.close("all")

    def test_returns_four_boxes(self):
        fig, ax = plot_ps_boxplot(self.df)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(ax.patches), 4)
        self.assertEqual(ax.get_ylabel(), "Propensity Score")
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))

    def test_weighting_reduces_median_gap(self):
        fig, ax = plot_ps_boxplot(self.df)
        # median = black horizontal line spanning exactly its box's width
        medians = []
        for patch in ax.patches:
            x0, x1 = (
                patch.get_path().vertices[:, 0].min(),
                patch.get_path().vertices[:, 0].max(),
            )
            for line in ax.lines:
                xs, ys = line.get_xdata(), line.get_ydata()
                if (
                    line.get_color() == "black"
                    and ys[0] == ys[-1]
                    and np.isclose(min(xs), x0)
                    and np.isclose(max(xs), x1)
                ):
                    medians.append(ys[0])
        self.assertEqual(len(medians), 4)
        unweighted_gap = abs(medians[1] - medians[0])
        weighted_gap = abs(medians[3] - medians[2])
        self.assertLess(weighted_gap, unweighted_gap)

    def test_att_weight_type_and_existing_ax(self):
        fig, ax = plt.subplots()
        out_fig, out_ax = plot_ps_boxplot(self.df, weight_type="ATT", fig=fig, ax=ax)
        self.assertIs(out_ax, ax)


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib is not installed")
class TestZipperPlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_coverage_in_legend(self):
        lower = np.array([-1.0, -1.0, 0.5, -1.0])
        upper = np.array([1.0, 1.0, 1.5, 1.0])
        fig, ax = plot_zipper(0.0, lower, upper)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels, ["Covers truth (75.0%)", "Misses truth (25.0%)"])
        self.assertEqual(len(ax.collections), 2)

    def test_per_replicate_truth(self):
        truth = np.array([0.0, 10.0])
        fig, ax = plot_zipper(truth, truth - 1, truth + 1)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels, ["Covers truth (100.0%)"])

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            plot_zipper(0.0, [1.0], [0.0])
        with self.assertRaises(ValueError):
            plot_zipper(0.0, [0.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            plot_zipper(0.0, [], [])
