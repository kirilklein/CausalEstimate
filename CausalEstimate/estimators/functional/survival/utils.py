"""
The IPCW survival curve is computed as defined here:
Cai, Weixin, and Mark J. van der Laan. "One-step targeted maximum likelihood estimation for time-to-event outcomes." Biometrics 76.3 (2020): 722-733.
"""

import numpy as np


def compute_IPCW_curve_at_t(
    patients_at_risk: int, ps: np.ndarray, survival_probs: np.ndarray
) -> float:
    """
    Compute the IPCW survival curve at a given time point.
    Args:
        patients_at_risk: Number of patients at risk at time t.
        ps: Propensity scores.
        survival_probs: Survival probabilities.
    """
    sum_ = np.sum(1 / (ps * survival_probs))
    return 1 / patients_at_risk * sum_
