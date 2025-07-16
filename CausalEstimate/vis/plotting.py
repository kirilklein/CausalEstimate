from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


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


def _process_single_dataset(
    df: pd.DataFrame,
    target_col: str,
    proba_col: str,
    n_bins: int,
    strategy: str,
    include_brier: bool,
    pos_label: Union[int, str],
) -> Dict[str, Any]:
    """
    Helper function to process a single dataset for calibration plotting.

    Returns a dictionary with all the processed data needed for plotting.
    """
    y_true = df[target_col].values
    y_prob = df[proba_col].values

    # Use sklearn's calibration_curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Calculate Brier score if needed
    brier_score = None
    if include_brier:
        brier_score = brier_score_loss(y_true, y_prob, pos_label=pos_label)

    # Calculate bin counts for annotations if needed
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    else:  # quantile strategy
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))

    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier_score": brier_score,
        "bin_counts": bin_counts,
    }


def plot_calibration(
    df: pd.DataFrame,
    proba_col: str = "probas",
    target_col: str = "targets",
    df2: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    labels: Tuple[str, str] = ("Model", "Comparison"),
    include_brier: bool = True,
    include_counts: bool = False,
    include_ideal: bool = True,
    markers: Tuple[str, str] = ("o", "s"),
    colors: Tuple[str, str] = ("b", "r"),
    alpha: float = 0.7,
    xlabel: str = "Mean Predicted Probability",
    ylabel: str = "Fraction of Positives",
    title: str = "Calibration Plot",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    pos_label: Union[int, str] = 1,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration curves for one or two datasets.

    Args:
        df (pd.DataFrame): DataFrame containing true labels and probability predictions.
        proba_col (str): Column name for predicted probabilities.
        target_col (str): Column name for true binary labels (0/1).
        df2 (pd.DataFrame, optional): Optional second DataFrame for comparison.
        n_bins (int): Number of bins for calibration curve.
        strategy (str): Binning strategy, 'uniform' creates equal-width bins,
                       'quantile' creates equal-populated bins.
        labels (tuple): Labels for each dataset (shown in legend with optional Brier scores).
        include_brier (bool): Whether to include Brier scores in the legend.
        include_counts (bool): Whether to display counts in each bin as text.
        include_ideal (bool): Whether to plot the ideal diagonal line.
        markers (tuple): Marker styles for the two curves.
        colors (tuple): Colors for the two curves.
        alpha (float): Transparency for markers.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        fig, ax: If provided, plot into them; otherwise create new figure/axes.
        figsize (tuple): Size of figure if we create a new one.
        pos_label: Label of the positive class for brier score calculation.

    Returns:
        (fig, ax): Figure and axes objects
    """
    # Create or reuse figure/axes
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is None:
        ax = fig.add_subplot(111)
    elif fig is None:
        fig = ax.figure

    # Process datasets
    datasets = [df]
    if df2 is not None:
        datasets.append(df2)

    results = []
    for i, dataset in enumerate(datasets):
        result = _process_single_dataset(
            df=dataset,
            target_col=target_col,
            proba_col=proba_col,
            n_bins=n_bins,
            strategy=strategy,
            include_brier=include_brier,
            pos_label=pos_label,
        )
        results.append(result)

    # Plot each dataset
    for i, result in enumerate(results):
        # Create label with optional Brier score
        label = labels[i]
        if include_brier and result["brier_score"] is not None:
            label = f"{label} (Brier = {result['brier_score']:.3f})"

        # Plot calibration curve
        ax.plot(
            result["prob_pred"],
            result["prob_true"],
            marker=markers[i],
            color=colors[i],
            alpha=alpha,
            label=label,
        )

        # Add count annotations if requested
        if include_counts:
            for x, y, count in zip(
                result["prob_pred"], result["prob_true"], result["bin_counts"]
            ):
                if count > 0:  # Only annotate if there are points in the bin
                    ax.annotate(
                        f"{count}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                        fontsize=8,
                    )

    # Plot ideal diagonal if requested
    if include_ideal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Ideal")

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_calibration_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    target_col: str = "targets",
    proba_col: str = "probas",
    n_bins: int = 10,
    strategy: str = "uniform",
    labels: Tuple[str, str] = ("Before", "After"),
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration curves for two datasets on the same axes.
    A convenience wrapper around plot_calibration.

    Args:
        df1, df2 (pd.DataFrame): DataFrames containing true labels and probability predictions.
        target_col (str): Column name for true binary labels (0/1).
        proba_col (str): Column name for predicted probabilities.
        n_bins (int): Number of bins for calibration curve.
        strategy (str): Binning strategy, 'uniform' creates equal-width bins,
                       'quantile' creates equal-populated bins.
        labels (tuple): Labels for each dataset (shown in legend with Brier scores).
        fig, ax: If provided, plot into them; otherwise create new figure/axes.
        figsize (tuple): Size of figure if we create a new one.
        **kwargs: Additional arguments passed to plot_calibration

    Returns:
        (fig, ax): Figure and axes objects
    """
    return plot_calibration(
        df=df1,
        df2=df2,
        proba_col=proba_col,
        target_col=target_col,
        n_bins=n_bins,
        strategy=strategy,
        labels=labels,
        fig=fig,
        ax=ax,
        figsize=figsize,
        **kwargs,
    )
