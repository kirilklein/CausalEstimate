import pandas as pd
from typing import List


def generate_bootstrap_samples(
    df: pd.DataFrame, n_bootstraps: int
) -> List[pd.DataFrame]:
    """
    Generate bootstrap samples using the standard non-parametric bootstrap method.

    This function creates multiple resampled datasets by sampling with replacement
    from the original dataset. Each bootstrap sample has the same number of
    observations as the original dataset, but some observations may be repeated
    while others may be omitted.

    The non-parametric bootstrap is useful for estimating the sampling distribution
    of a statistic and for computing confidence intervals or standard errors when
    the underlying distribution is unknown or complex.

    Args:
        df: The original dataset to bootstrap from.
        n_bootstraps: The number of bootstrap samples to generate.

    Returns:
        A list containing n_bootstraps number of resampled
        DataFrames, each with the same shape as the input df.
    """
    n = len(df)
    return [df.sample(n=n, replace=True) for _ in range(n_bootstraps)]
