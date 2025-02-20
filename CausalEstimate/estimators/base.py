# CausalEstimate/estimators/base.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseEstimator(ABC):
    def __init__(
        self,
        effect_type: str = "ATE",
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        ps_col: str = "ps",
        **kwargs,
    ):
        """
        Base class for all estimators.
        - effect_type: e.g. "ATE", "ATT", ...
        - treatment_col, outcome_col, ps_col: universal column names
        - kwargs: any additional method-specific settings or toggles
        """
        self.effect_type = effect_type
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.ps_col = ps_col
        self.kwargs = kwargs  # if child classes want extra toggles

    @abstractmethod
    def compute_effect(self, df: pd.DataFrame) -> float:
        """
        Compute the causal effect from the given dataframe.
        The columns to use are already known from the constructor.
        This method is to be implemented by child classes.
        """
        pass
