from CausalEstimate.estimators.functional.aipw import compute_aipw_ate
import pandas as pd


class IPW:
    def __init__(self, effect_type="ATE", **kwargs):
        self.effect_type = effect_type
        self.kwargs = kwargs

    def compute_effect(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ps_col: str,
        predicted_outcome_treated_col: str,
        predicted_outcome_control_col: str,
    ) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE, ATT, RR, RRT
        """

        A = df[treatment_col]
        Y = df[outcome_col]
        ps = df[ps_col]

        if self.effect_type == "ATE":
            return compute_aipw_ate(A, Y, ps)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
