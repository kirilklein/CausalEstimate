from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_outcome_proba_dist(
    df: pd.DataFrame,
    outcome_proba_col: str,
    treatment_col: str,
    xlabel: str = "Predicted Outcome Probability",
    title: str = "Outcome Probability Distribution",
    bin_edges: np.ndarray = None,
    normalize: bool = False,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 6),
):
    """
    Plot a predicted-outcome probability distribution for treatment vs. control groups.
    E.g., if 'outcome_proba_col' stores model-predicted probabilities.
    """
    return plot_hist_by_groups(
        df=df,
        value_col=outcome_proba_col,
        group_col=treatment_col,
        group_values=(0, 1),
        group_labels=("Control", "Treatment"),
        bin_edges=bin_edges,
        normalize=normalize,
        xlabel=xlabel,
        title=title,
        fig=fig,
        ax=ax,
        figsize=figsize,
    )


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
    """
    return plot_hist_by_groups(
        df=df,
        value_col=ps_col,
        group_col=treatment_col,
        group_values=(0, 1),
        group_labels=("Control", "Treatment"),
        bin_edges=bin_edges,
        normalize=normalize,
        xlabel=xlabel,
        title=title,
        fig=fig,
        ax=ax,
        figsize=figsize,
    )


def plot_hist_by_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_values=(0, 1),
    group_labels=("Group 0", "Group 1"),
    bin_edges=None,
    normalize: bool = False,
    xlabel: str = None,
    title: str = None,
    alpha: float = 0.5,
    colors=("b", "r"),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    A generic helper that plots a histogram of 'value_col' for two groups
    defined by 'group_col', e.g. group_col=0 vs. group_col=1.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        value_col (str): The column whose distribution we want to plot.
        group_col (str): The column that indicates group membership.
        group_values (tuple): The two distinct values used to split the DataFrame.
        group_labels (tuple): Labels for legend (e.g. "Control", "Treatment").
        bin_edges (array): The bin edges for histogram. If None, defaults to 50 bins from 0..1
        normalize (bool): Whether to normalize the histogram (density=True).
        xlabel (str): X-axis label.
        title (str): Plot title.
        alpha (float): Transparency for the histogram overlay.
        colors (tuple): Colors for the two histograms.
        fig, ax: If provided, plot into them; otherwise create new figure/axes.
        figsize (tuple): Size of figure if we create a new one.

    Returns:
        (fig, ax)
    """
    # create or reuse figure/axes
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is None:
        ax = fig.add_subplot(111)
    elif fig is None:
        raise ValueError("fig and ax cannot both be None")

    # default bins
    if bin_edges is None:
        bin_edges = np.linspace(0, 1, 51)  # 50 bins in [0,1]

    # group 0
    mask0 = df[group_col] == group_values[0]
    ax.hist(
        df.loc[mask0, value_col],
        bins=bin_edges,
        alpha=alpha,
        label=group_labels[0],
        color=colors[0],
        density=normalize,
    )

    # group 1
    mask1 = df[group_col] == group_values[1]
    ax.hist(
        df.loc[mask1, value_col],
        bins=bin_edges,
        alpha=alpha,
        label=group_labels[1],
        color=colors[1],
        density=normalize,
    )

    ax.set_xlabel(xlabel if xlabel else value_col)
    ax.set_ylabel("Count" if not normalize else "Density")
    if title:
        ax.set_title(title)
    ax.legend()

    return fig, ax
