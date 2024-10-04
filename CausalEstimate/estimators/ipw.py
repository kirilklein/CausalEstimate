import pandas as pd

from CausalEstimate.core.registry import register_estimator
from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_ate_stabilized,
    compute_ipw_att,
    compute_ipw_risk_ratio,
    compute_ipw_risk_ratio_treated,
)
from CausalEstimate.utils.checks import check_inputs


@register_estimator
class IPW(BaseEstimator):
    def __init__(self, effect_type="ATE", **kwargs):
        super().__init__(effect_type=effect_type, **kwargs)

    def compute_effect(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ps_col: str,
        **kwargs,
    ) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE, ATT, RR, RRT
        """

        A = df[treatment_col]
        Y = df[outcome_col]
        ps = df[ps_col]
        check_inputs(A, Y, ps)
        if self.effect_type == "ATE":
            if self.kwargs.get("stabilized", False):
                return compute_ipw_ate_stabilized(A, Y, ps)
            else:
                return compute_ipw_ate(A, Y, ps)
        elif self.effect_type == "ATT":
            return compute_ipw_att(A, Y, ps)
        elif self.effect_type == "RR":
            return compute_ipw_risk_ratio(A, Y, ps)
        elif self.effect_type == "RRT":
            return compute_ipw_risk_ratio_treated(A, Y, ps)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
