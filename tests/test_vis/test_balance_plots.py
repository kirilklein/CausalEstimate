import unittest

import numpy as np
import pandas as pd

from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from CausalEstimate.diagnostics import compute_balance_table
    from CausalEstimate.vis.plotting import plot_love, plot_weight_dist

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
