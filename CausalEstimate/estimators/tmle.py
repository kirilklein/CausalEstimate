from typing import Optional, Tuple

import numpy as np
import pandas as pd

from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.tmle import compute_tmle_ate, compute_tmle_rr
from CausalEstimate.estimators.functional.tmle_att import compute_tmle_att
from CausalEstimate.utils.checks import check_inputs, check_required_columns
from CausalEstimate.utils.constants import (
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
    BINARY_OUTCOME_EFFECTS,
    CI95_LOWER,
    CI95_UPPER,
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
    INITIAL_EFFECT,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
    STD_ERR,
)

# Result keys that are differences (scale by range) vs. means (scale and shift)
_DIFF_KEYS = (EFFECT, INITIAL_EFFECT, ADJUSTMENT_treated, ADJUSTMENT_untreated)
_CI_KEYS = (STD_ERR, CI95_LOWER, CI95_UPPER)
_MEAN_KEYS = (
    EFFECT_treated,
    EFFECT_untreated,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
)


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
        clip_percentile: float = 1,
        eps: float = 1e-9,
        y_bounds: Optional[Tuple[float, float]] = None,
    ):
        """
        Targeted Maximum Likelihood Estimation (TMLE) estimator.

        Binary outcomes use the logistic fluctuation directly. A continuous
        outcome (ATE/ATT/ARR only) is rescaled to [0, 1] with ``y_bounds``
        (default: observed min/max of the outcome), targeted on that scale,
        and the results are mapped back (Gruber & van der Laan, 2010).

        Args:
            effect_type: Type of causal effect to estimate
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            ps_col: Name of propensity score column
            probas_col: Name of predicted probabilities column
            probas_t1_col: Name of predicted probabilities under treatment column
            probas_t0_col: Name of predicted probabilities under control column
            clip_percentile: Upper percentile for clipping, in (0, 1]. Default 1 (no clipping).
            eps: Small constant for numerical stability in denominators
            y_bounds: (min, max) of a continuous outcome. Predictions are
                clipped to these bounds. Ignored for binary outcomes.
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
        self.clip_percentile = clip_percentile
        self.eps = eps
        self.y_bounds = y_bounds

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

        is_binary = self.effect_type in BINARY_OUTCOME_EFFECTS or (
            self.y_bounds is None and set(np.unique(Y)) <= {0, 1}
        )
        check_inputs(
            A, Y, ps, Yhat=Yhat, Y1_hat=Y1_hat, Y0_hat=Y0_hat, binary_outcome=is_binary
        )

        if not (0 < self.clip_percentile <= 1):
            raise ValueError("clip_percentile must be in (0, 1].")

        if is_binary:
            return self._targeted_effect(A, Y, ps, Y0_hat, Y1_hat, Yhat)

        lo, hi = self.y_bounds if self.y_bounds is not None else (Y.min(), Y.max())
        if not hi > lo:
            raise ValueError("y_bounds must satisfy min < max.")
        if not ((Y >= lo) & (Y <= hi)).all():
            raise ValueError("Outcome contains values outside y_bounds.")
        scale = hi - lo

        def to_unit(x):
            return np.clip((x - lo) / scale, 0, 1)

        result = self._targeted_effect(
            A, to_unit(Y), ps, to_unit(Y0_hat), to_unit(Y1_hat), to_unit(Yhat)
        )
        for k in _DIFF_KEYS + _CI_KEYS:
            if k in result:
                result[k] = result[k] * scale
        for k in _MEAN_KEYS:
            if k in result:
                result[k] = lo + result[k] * scale
        return result

    def _targeted_effect(self, A, Y, ps, Y0_hat, Y1_hat, Yhat) -> dict:
        if self.effect_type in ["ATE", "ARR"]:
            return compute_tmle_ate(
                A,
                Y,
                ps,
                Y0_hat,
                Y1_hat,
                Yhat,
                clip_percentile=self.clip_percentile,
                eps=self.eps,
            )
        elif self.effect_type == "ATT":
            return compute_tmle_att(
                A,
                Y,
                ps,
                Y0_hat,
                Y1_hat,
                Yhat,
                clip_percentile=self.clip_percentile,
                eps=self.eps,
            )
        elif self.effect_type == "RR":
            return compute_tmle_rr(
                A,
                Y,
                ps,
                Y0_hat,
                Y1_hat,
                Yhat,
                clip_percentile=self.clip_percentile,
                eps=self.eps,
            )
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
