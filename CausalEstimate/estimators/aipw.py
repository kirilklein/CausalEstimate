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
        **kwargs,
    ):
        super().__init__(
            effect_type=effect_type,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            **kwargs,
        )
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
