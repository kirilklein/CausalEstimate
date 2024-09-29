from CausalEstimate.matching.matching import match_optimal
from CausalEstimate.estimators.functional.matching import compute_matching_ate


class MATCHING:
    def __init__(self, effect_type="ATE", **kwargs):
        self.effect_type = effect_type
        self.kwargs = kwargs

    def compute_effect(self, df, treatment_col, outcome_col, ps_col) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE, RR, OR
        """

        Y = df[outcome_col].values

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
