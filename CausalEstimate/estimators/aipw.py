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
        check_required_columns(
            df,
            [
                self.probas_t1_col,
                self.probas_t0_col,
            ],
        )
        A = df[self.treatment_col]
        Y = df[self.outcome_col]
        ps = df[self.ps_col]
        Y1_hat = df[self.probas_t1_col]
        Y0_hat = df[self.probas_t0_col]

        check_inputs(A, Y, ps, Y1_hat=Y1_hat, Y0_hat=Y0_hat)

        if self.effect_type in ["ATE", "ARR"]:
            return compute_aipw_ate(A, Y, ps, Y0_hat, Y1_hat)
        elif self.effect_type == "ATT":
            return compute_aipw_att(A, Y, ps, Y0_hat)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
