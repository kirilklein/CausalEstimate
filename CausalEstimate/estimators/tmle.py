# CausalEstimate/estimators/tmle.py
import pandas as pd

from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.tmle import compute_tmle_ate, compute_tmle_rr
from CausalEstimate.estimators.functional.tmle_att import compute_tmle_att
from CausalEstimate.utils.checks import check_inputs, check_required_columns


class TMLE(BaseEstimator):
    def __init__(
        self,
        effect_type: str = "ATE",
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        ps_col: str = "ps",
        probas_col: str = "probas",
        probas_t1_col: str = "probas_t1",
        probas_t0_col: str = "probas_t0",
        stabilized: bool = False,
    ):
        """
        Targeted Maximum Likelihood Estimation (TMLE) estimator.

        Args:
            effect_type: Type of causal effect to estimate
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            ps_col: Name of propensity score column
            probas_col: Name of predicted probabilities column
            probas_t1_col: Name of predicted probabilities under treatment column
            probas_t0_col: Name of predicted probabilities under control column
            stabilized: Whether to use stabilized weights
        """
        # Initialize base class with core parameters
        super().__init__(
            effect_type=effect_type,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
        )

        # TMLE-specific parameters
        self.probas_col = probas_col
        self.probas_t1_col = probas_t1_col
        self.probas_t0_col = probas_t0_col
        self.stabilized = stabilized

    def _compute_effect(self, df: pd.DataFrame) -> dict:
        """Compute causal effect using TMLE."""
        # Check TMLE-specific columns
        check_required_columns(
            df,
            [self.probas_col, self.probas_t1_col, self.probas_t0_col],
        )

        A, Y, ps, Yhat, Y1_hat, Y0_hat = self._get_numpy_arrays(
            df,
            [
                self.treatment_col,
                self.outcome_col,
                self.ps_col,
                self.probas_col,
                self.probas_t1_col,
                self.probas_t0_col,
            ],
        )

        check_inputs(A, Y, ps, Yhat=Yhat, Y1_hat=Y1_hat, Y0_hat=Y0_hat)

        if self.effect_type in ["ATE", "ARR"]:
            return compute_tmle_ate(
                A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=self.stabilized
            )
        elif self.effect_type == "ATT":
            return compute_tmle_att(
                A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=self.stabilized
            )
        elif self.effect_type == "RR":
            return compute_tmle_rr(
                A, Y, ps, Y0_hat, Y1_hat, Yhat, stabilized=self.stabilized
            )
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
