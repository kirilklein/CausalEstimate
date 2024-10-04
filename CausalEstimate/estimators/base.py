import pandas as pd


class BaseEstimator:
    def __init__(self, effect_type: str, **kwargs):
        self.effect_type = effect_type
        self.kwargs = kwargs  # Store additional keyword arguments if needed

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
    ):
        raise NotImplementedError("This method should be implemented by subclasses.")
