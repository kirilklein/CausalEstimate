from CausalEstimate.estimators.functional.tmle import compute_tmle_ate
import pandas as pd


class TMLE:
    def __init__(self, effect_type="ATE", **kwargs):
        self.effect_type = effect_type
        self.kwargs = kwargs

    def compute_effect(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ps_col: str,
        predicted_outcome_col: str,
        predicted_outcome_treated_col: str,
        predicted_outcome_control_col: str,
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

        if self.effect_type == "ATE":
            return compute_tmle_ate(A, Y, ps, Y0_hat, Y1_hat, Yhat)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
