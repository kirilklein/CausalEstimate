import numpy as np
import pandas as pd


def is_sparse(matrix: np.array, threshold: float = 0.5) -> bool:
    """
    Checks whether a matrix is sparse.
    Args:
        matrix (np.array): Input matrix.
    Returns:
        bool: True if matrix is sparse, False otherwise.
    """
    return np.count_nonzero(matrix) < threshold * matrix.size


def check_ps_validity(df: pd.DataFrame, ps_col: str) -> None:
    """
    Check if the propensity scores are valid.
    Propensity scores should be between 0 and 1.
    """
    if not df[ps_col].between(0, 1).all():
        raise ValueError("Propensity scores should be between 0 and 1.")
