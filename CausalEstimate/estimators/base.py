# CausalEstimate/estimators/base.py
import warnings
from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


class BaseEstimator(ABC):
    VALID_EFFECT_TYPES = {"ATE", "ARR", "ATT", "ATC", "RR", "RRT"}  #
    PS_EXTREME_THRESHOLD = 0.01  # Configurable threshold for extreme PS warning
    MIN_GROUP_SIZE = 10  # Configurable minimum group size

    def __init__(
        self,
        effect_type: Literal[
            "ATE", "ARR", "ATT", "ATC", "RR", "RRT"
        ] = "ATE",  # if binary it is equivalent
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        ps_col: str = "ps",
        **kwargs,
    ):
        """
        Base class for all estimators.
        - effect_type: e.g. "ATE", "ATT", ... (ARR absolute risk reduction is equivalent to ATE if binary)
        - treatment_col, outcome_col, ps_col: universal column names
        - kwargs: any additional method-specific settings or toggles
        """
        self.effect_type = effect_type
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.ps_col = ps_col
        self.kwargs = kwargs  # if child classes want extra toggles

    def compute_effect(self, df: pd.DataFrame) -> dict:
        """
        Compute the causal effect from the given dataframe.
        The columns to use are already known from the constructor.
        This method is to be implemented by child classes.
        """
        self._validate_input_df(df)
        return self._compute_effect(df)

    @abstractmethod
    def _compute_effect(self, df: pd.DataFrame) -> dict:
        """
        Concrete implementation to be provided by child classes.
        Input df is guaranteed to be validated.
        """
        pass

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """
        Main validation method that orchestrates all validation checks.

        Raises:
            TypeError: If input is not a pandas DataFrame
            ValueError: If any validation fails
        Warns:
            RuntimeWarning: For concerning but non-fatal issues
        """
        self._validate_df_type(df)
        self._validate_columns_exist(df)

        # Extract columns once for efficiency
        treatment = df[self.treatment_col]
        ps = df[self.ps_col]

        outcome = df[self.outcome_col]

        self._validate_propensity_scores(ps)
        self._validate_treatment_values(treatment)
        self._validate_group_sizes(treatment)
        self._validate_missing_values(df)

        if self.effect_type in {
            "RR",
            "RRT",
            "ARR",
        }:  # should be binary for ratio-based estimands
            self._validate_binary_outcome(outcome)

    def _validate_df_type(self, df: pd.DataFrame) -> None:
        """Validate input is a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

    def _validate_columns_exist(self, df: pd.DataFrame) -> None:
        """Check all required columns are present."""
        required_cols = {self.treatment_col, self.outcome_col, self.ps_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _validate_propensity_scores(self, ps: pd.Series) -> None:
        """Validate propensity scores range and check for extreme values."""
        # Check basic range
        if not ps.between(0, 1).all():
            raise ValueError("Propensity scores must be between 0 and 1")

        # Check for extreme values
        extreme_ps = (ps < self.PS_EXTREME_THRESHOLD) | (
            ps > (1 - self.PS_EXTREME_THRESHOLD)
        )
        if extreme_ps.any():
            n_extreme = extreme_ps.sum()
            warnings.warn(
                f"Found {n_extreme} ({n_extreme/len(ps):.1%}) propensity scores "
                f"outside [{self.PS_EXTREME_THRESHOLD}, {1-self.PS_EXTREME_THRESHOLD}]",
                RuntimeWarning,
            )

    def _validate_treatment_values(self, treatment: pd.Series) -> None:
        """Validate treatment is binary."""
        unique_treatments = set(treatment.unique())
        if not unique_treatments.issubset({0, 1}):
            raise ValueError(
                f"Treatment must be binary (0 or 1), found values: {unique_treatments}"
            )

    def _validate_group_sizes(self, treatment: pd.Series) -> None:
        """Check for sufficient sample sizes in treatment groups."""
        n_treated = (treatment == 1).sum()
        n_control = (treatment == 0).sum()

        if n_treated < self.MIN_GROUP_SIZE or n_control < self.MIN_GROUP_SIZE:
            warnings.warn(
                f"Small group sizes detected: treated={n_treated}, control={n_control}. "
                f"Minimum recommended size is {self.MIN_GROUP_SIZE}.",
                RuntimeWarning,
            )

    def _validate_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in required columns."""
        required_cols = {self.treatment_col, self.outcome_col, self.ps_col}
        for col in required_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                raise ValueError(f"Found {n_missing} missing values in column '{col}'")

    def _validate_binary_outcome(self, outcome: pd.Series) -> None:
        """Validate outcome is binary for ratio-based estimands."""
        if not outcome.between(0, 1).all():
            raise ValueError(
                f"Outcome must be binary for {self.effect_type}, "
                f"found values outside [0,1]"
            )
