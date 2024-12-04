"""
The IPCW survival curve is computed as defined here:
Cai, Weixin, and Mark J. van der Laan. "One-step targeted maximum likelihood estimation for time-to-event outcomes." Biometrics 76.3 (2020): 722-733.
"""
import numpy as np
import pandas as pd

from CausalEstimate.core.registry import register_estimator
from CausalEstimate.estimators.base import BaseEstimatorSurvival
from CausalEstimate.estimators.functional.survival.utils import \
    compute_IPCW_curve_at_t


@register_estimator
class IPCW(BaseEstimatorSurvival):
    def __init__(self, effect_type="survival", **kwargs):
        super().__init__(effect_type=effect_type, **kwargs)

    def compute_effect(
        self,
        df: pd.DataFrame,
        ps_col: str,
        survival_prob_col: str,
        time_col: str,
    ) -> np.ndarray:
        """
        Compute the IPCW survival curve.
        """
        # Apply IPCW logic, similar to your provided code:
        max_time = df[time_col].max()
        survival_curve = []
        
        for t in range(1, max_time + 1):
            # Select patients at risk and compute the curve at each time point
            at_risk = df[df[time_col] <= t]
            ps_values = at_risk[ps_col]
            survival_probs = at_risk[survival_prob_col]
            survival_curve_at_t = compute_IPCW_curve_at_t(len(at_risk), ps_values, survival_probs)
            survival_curve.append(survival_curve_at_t)
            
        return np.array(survival_curve)