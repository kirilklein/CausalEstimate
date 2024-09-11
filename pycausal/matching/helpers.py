import numpy as np


def is_sparse(matrix: np.array, threshold: float = 0.5) -> bool:
    """
    Checks whether a matrix is sparse.
    Args:
        matrix (np.array): Input matrix.
    Returns:
        bool: True if matrix is sparse, False otherwise.
    """
    return np.count_nonzero(matrix) < threshold * matrix.size
