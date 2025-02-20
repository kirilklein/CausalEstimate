from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.matching.matching import match_optimal
from CausalEstimate.utils.checks import check_inputs, check_required_columns
import warnings


class Matching(BaseEstimator):
    def __init__(
        self,
        effect_type="ATE",
        treatment_col="treatment",
        outcome_col="outcome",
        ps_col="ps",
        **kwargs,
    ):
        super().__init__(effect_type=effect_type, **kwargs)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.ps_col = ps_col
        self.kwargs = kwargs

    def compute_effect(
        self,
        df,
    ) -> float:
        """
        Compute the effect using matching.
        Available effect types: ATE

        Effect computation for matched groups is different from the functional IPW.
        The matching itself influences the selected population, and thus the type of effect computed.
        E.g. when chosing ALL the trated and matching untreated to them, we get the ATT.
        However, when setting a caliper, we get a differnt population which is neither ATT nor ATE.[1]
        For ATE we can use full matching [2]

        [1] Greifer, Estimating Effects After Matching (https://cran.r-project.org/web/packages/MatchIt/vignettes/estimating-effects.html)
        [2] Stuart, et. al. "Using full matching to estimate causal effects in nonexperimental studies:
                examining the relationship between adolescent marijuana use and adult outcomes."
                Developmental psychology 44.2 (2008): 395.
        """
        check_required_columns(df, [self.treatment_col, self.outcome_col, self.ps_col])
        Y = df[self.outcome_col]
        check_inputs(df[self.treatment_col], Y, df[self.ps_col])
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df["index"] = df.index  # temporary index column
        matched = match_optimal(
            df,
            treatment_col=self.treatment_col,
            ps_col=self.ps_col,
            pid_col="index",
            **self.kwargs,
        )
        if self.effect_type == "ATE":
            warnings.warn(
                "This is strictly speaking not ATE if we used a caliper or other matching methods. But can be interpreted as such."
            )
            return compute_matching_ate(Y, matched)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
