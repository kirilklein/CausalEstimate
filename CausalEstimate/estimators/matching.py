from CausalEstimate.core.registry import register_estimator
from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.matching.matching import match_optimal
from CausalEstimate.utils.checks import check_inputs


@register_estimator
class MATCHING(BaseEstimator):
    def __init__(self, effect_type="ATE", **kwargs):
        super().__init__(effect_type=effect_type, **kwargs)

    def compute_effect(self, df, treatment_col, outcome_col, ps_col) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE, RR, OR
        """

        Y = df[outcome_col].values
        check_inputs(df[treatment_col], Y, df[ps_col])
        df["index"] = df.index  # temporary index column
        matched = match_optimal(
            df,
            treatment_col=treatment_col,
            ps_col=ps_col,
            pid_col="index",
            **self.kwargs,
        )
        if self.effect_type == "ATE":
            return compute_matching_ate(Y, matched)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
