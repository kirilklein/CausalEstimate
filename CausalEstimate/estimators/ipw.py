import pandas as pd

from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_ate_stabilized,
    compute_ipw_att,
    compute_ipw_risk_ratio,
    compute_ipw_risk_ratio_treated,
)
from CausalEstimate.utils.checks import check_inputs, check_required_columns


class IPW(BaseEstimator):
    def __init__(
        self,
        effect_type="ATE",
        treatment_col="treatment",
        outcome_col="outcome",
        ps_col="ps",
        **kwargs,
    ):
        super().__init__(effect_type=effect_type, **kwargs)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.ps_col = ps_col
        self.kwargs = kwargs

    def compute_effect(self, df: pd.DataFrame) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE, ATT, RR, RRT
        """
        check_required_columns(df, [self.treatment_col, self.outcome_col, self.ps_col])
        A = df[self.treatment_col]
        Y = df[self.outcome_col]
        ps = df[self.ps_col]
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
