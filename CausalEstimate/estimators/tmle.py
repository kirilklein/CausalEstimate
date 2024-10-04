import pandas as pd

from CausalEstimate.core.registry import register_estimator
from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.tmle import compute_tmle_ate
from CausalEstimate.utils.checks import check_inputs


@register_estimator
class TMLE(BaseEstimator):
    def __init__(self, effect_type="ATE", **kwargs):
        super().__init__(effect_type=effect_type, **kwargs)

    def compute_effect(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ps_col: str,
        predicted_outcome_col: str,
        predicted_outcome_treated_col: str,
        predicted_outcome_control_col: str,
        **kwargs,
    ) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE
        """

        A = df[treatment_col]
        Y = df[outcome_col]
        ps = df[ps_col]
        Y0_hat = df[predicted_outcome_control_col]
        Y1_hat = df[predicted_outcome_treated_col]
        Yhat = df[predicted_outcome_col]
        check_inputs(A, Y, ps, Y1_hat=Y1_hat, Y0_hat=Y0_hat, Yhat=Yhat)
        if self.effect_type == "ATE":
            return compute_tmle_ate(A, Y, ps, Y0_hat, Y1_hat, Yhat)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
