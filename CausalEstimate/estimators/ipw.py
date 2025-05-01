import pandas as pd

from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_ate_stabilized,
    compute_ipw_att,
    compute_ipw_risk_ratio,
    compute_ipw_risk_ratio_treated,
)


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

    def _compute_effect(self, df: pd.DataFrame) -> dict:
        """
        Calculates the specified causal effect using inverse probability weighting (IPW).

        Extracts treatment, outcome, and propensity score arrays from the input DataFrame and computes the effect based on the configured effect type. Supports average treatment effect (ATE), average treatment effect on the treated (ATT), risk ratio (RR), and risk ratio for the treated (RRT). For ATE or ARR, uses stabilized weights if specified.

        Args:
            df: Input DataFrame containing treatment, outcome, and propensity score columns.

        Returns:
            A dictionary with the computed effect estimate.

        Raises:
            ValueError: If the effect type is not supported.
        """
        A, Y, ps = self._get_numpy_arrays(
            df, [self.treatment_col, self.outcome_col, self.ps_col]
        )
        if self.effect_type in ["ATE", "ARR"]:
            if self.kwargs.get("stabilized", False):
                return compute_ipw_ate_stabilized(A, Y, ps)
            else:
                return compute_ipw_ate(A, Y, ps)
        elif self.effect_type in ["ATT"]:
            return compute_ipw_att(A, Y, ps)
        elif self.effect_type == "RR":
            return compute_ipw_risk_ratio(A, Y, ps)
        elif self.effect_type == "RRT":
            return compute_ipw_risk_ratio_treated(A, Y, ps)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
