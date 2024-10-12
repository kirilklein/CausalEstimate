import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CausalEstimate.utils.utils import get_treated_ps, get_untreated_ps


def plot_propensity_score_dist(
    df: pd.DataFrame,
    ps_col: str,
    treatment_col: str,
    xlabel: str = "Propensity Score",
    title: str = "Propensity Score Distribution",
    bin_edges: np.ndarray = None,
    normalize: bool = False,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 6),
):
    """
    Plot a propensity score distribution for treatment and control groups.

    Parameters:
    -----------
    df : DataFrame containing the data.
    ps_col : Name of the column containing propensity scores.
    treatment_col : Name of the column indicating treatment status (0 for control, 1 for treatment).
    xlabel : Default is 'Propensity Score'.
    bin_edges : bin edges for histogram.
    normalize : Whether to normalize the histogram. Default is False.
    title : Default is 'Propensity Score Distribution'.
    fig : If provided, the plot will be added to this figure.
    ax : If provided, the plot will be added to this axis.
    figsize : Figure size. Default is (10, 6).

    Returns:
    --------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is None:
        ax = fig.add_subplot(111)
    elif fig is None:
        raise ValueError("fig and ax cannot both be None")

    if bin_edges is None:
        bin_edges = np.linspace(0, 1, 51)  # Creates 50 bins between 0 and 1

    ax.hist(
        get_untreated_ps(df, treatment_col, ps_col),
        bins=bin_edges,
        alpha=0.5,
        label="Control",
        color="b",
        density=normalize,
    )
    ax.hist(
        get_treated_ps(df, treatment_col, ps_col),
        bins=bin_edges,
        alpha=0.5,
        label="Treatment",
        color="r",
        density=normalize,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count" if not normalize else "Density")
    ax.set_title(title)
    ax.legend()

    return fig, ax
