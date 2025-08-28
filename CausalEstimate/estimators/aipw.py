# CausalEstimate/estimators/aipw.py

import pandas as pd

from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.aipw import compute_aipw_ate, compute_aipw_att
from CausalEstimate.utils.checks import check_inputs, check_required_columns


class AIPW(BaseEstimator):
    def __init__(
        self,
        effect_type: str = "ATE",
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        ps_col: str = "ps",
        probas_t1_col: str = "probas_t1",
        probas_t0_col: str = "probas_t0",
    ):
        """
        Augmented Inverse Probability Weighting (AIPW) estimator.

        Args:
            effect_type: Type of causal effect to estimate
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            ps_col: Name of propensity score column
            probas_t1_col: Name of predicted probabilities under treatment column
            probas_t0_col: Name of predicted probabilities under control column
        """
        # Initialize base class with core parameters
        super().__init__(
            effect_type=effect_type,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
        )

        # AIPW-specific parameters
        self.probas_t1_col = probas_t1_col
        self.probas_t0_col = probas_t0_col

    def _compute_effect(self, df: pd.DataFrame) -> dict:
        """
        Computes the causal effect estimate using the Augmented Inverse Probability Weighting (AIPW) method.

        Depending on the specified effect type, calculates the average treatment effect (ATE), average risk reduction (ARR), or average treatment effect on the treated (ATT) using the provided DataFrame. Requires columns for treatment assignment, observed outcome, propensity score, and predicted potential outcomes under treatment and control.

        Args:
            df: Input DataFrame containing the necessary columns for effect estimation.

        Returns:
            A dictionary with the estimated effect and related statistics.

        Raises:
            ValueError: If the specified effect type is not supported.
        """
        # Check AIPW-specific columns
        check_required_columns(
            df,
            [
                self.probas_t1_col,
                self.probas_t0_col,
            ],
        )

        A, Y, ps, Y1_hat, Y0_hat = self._get_numpy_arrays(
            df,
            [
                self.treatment_col,
                self.outcome_col,
                self.ps_col,
                self.probas_t1_col,
                self.probas_t0_col,
            ],
        )

        check_inputs(A, Y, ps, Y1_hat=Y1_hat, Y0_hat=Y0_hat)

        if self.effect_type in ["ATE", "ARR"]:
            return compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat)
        elif self.effect_type == "ATT":
            return compute_aipw_att(A, Y, ps, Y0_hat)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
