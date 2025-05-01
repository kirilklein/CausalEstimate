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
        **kwargs,
    ):
        super().__init__(
            effect_type=effect_type,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            **kwargs,
        )
        self.probas_col = probas_col
        self.probas_t1_col = probas_t1_col
        self.probas_t0_col = probas_t0_col

    def _compute_effect(self, df: pd.DataFrame) -> dict:
        # additional checks required for TMLE
        """
        Computes the specified causal effect estimate using Targeted Maximum Likelihood Estimation.

        Validates required columns and input arrays, then calculates the effect based on the estimator's effect type ("ATE", "ARR", "ATT", or "RR") using the appropriate TMLE computation function.

        Args:
            df: A pandas DataFrame containing treatment, outcome, propensity score, and predicted probability columns.

        Returns:
            A dictionary with the estimated effect and related statistics.

        Raises:
            ValueError: If the specified effect type is not supported.
        """
        check_required_columns(
            df,
            [
                self.probas_col,
                self.probas_t1_col,
                self.probas_t0_col,
            ],
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
            return compute_tmle_ate(A, Y, ps, Y0_hat, Y1_hat, Yhat)
        elif self.effect_type == "ATT":
            return compute_tmle_att(A, Y, ps, Y0_hat, Y1_hat, Yhat)
        elif self.effect_type == "RR":
            return compute_tmle_rr(A, Y, ps, Y0_hat, Y1_hat, Yhat)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
