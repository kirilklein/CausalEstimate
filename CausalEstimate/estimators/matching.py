import warnings

import numpy as np

from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.estimators.functional.matching import compute_matching_ate
from CausalEstimate.matching.matching import match_eager, match_optimal
from CausalEstimate.utils.constants import OUTCOME_COL, PS_COL, TREATMENT_COL


class Matching(BaseEstimator):
    def __init__(
        self,
        effect_type: str = "ATE",
        treatment_col: str = TREATMENT_COL,
        outcome_col: str = OUTCOME_COL,
        ps_col: str = PS_COL,
        match_optimal: bool = True,
        n_controls: int = 1,
        caliper: float = None,
        strict: bool = True,
    ):
        """
        Propensity Score Matching estimator.

        Args:
            effect_type: Type of causal effect to estimate
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            ps_col: Name of propensity score column
            match_optimal: Whether to use optimal matching (True) or greedy matching (False)
            n_controls: Number of controls to match for each treated individual
            caliper: Maximum allowable distance (propensity score difference) for matching
            strict: If True (and using greedy matching), raise error if any treated subject
                   cannot be matched. If False, skip unmatched subjects.
        """
        # Initialize base class with core parameters
        super().__init__(
            effect_type=effect_type,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
        )

        # Matching-specific parameters
        self.match_optimal = match_optimal
        self.n_controls = n_controls
        self.caliper = caliper
        self.strict = strict

    def _compute_effect(self, df) -> dict:
        """
        Compute the effect using matching.
        Available effect types: ATE

        Effect computation for matched groups is different from the functional IPW.
        The matching itself influences the selected population, and thus the type of effect computed.
        E.g. when choosing ALL the treated and matching untreated to them, we get the ATT.
        However, when setting a caliper, we get a different population which is neither ATT nor ATE.[1]
        For ATE we can use full matching [2]

        [1] Greifer, Estimating Effects After Matching (https://cran.r-project.org/web/packages/MatchIt/vignettes/estimating-effects.html)
        [2] Stuart, et. al. "Using full matching to estimate causal effects in nonexperimental studies:
                examining the relationship between adolescent marijuana use and adult outcomes."
                Developmental psychology 44.2 (2008): 395.
        """
        Y = df[self.outcome_col]
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning

        # Add tiny random noise to propensity scores to break ties
        eps = 1e-10  # Small enough to not meaningfully affect matching
        df[self.ps_col] = df[self.ps_col] + np.random.uniform(-eps, eps, size=len(df))
        # Ensure PS stays in [0,1] range
        df[self.ps_col] = df[self.ps_col].clip(0, 1)

        df["index"] = range(
            len(df)
        )  # This ensures unique PIDs even with bootstrap samples

        if self.match_optimal:
            matched = match_optimal(
                df,
                treatment_col=self.treatment_col,
                ps_col=self.ps_col,
                pid_col="index",
                n_controls=self.n_controls,
                caliper=self.caliper,
            )
        else:
            matched = match_eager(
                df,
                treatment_col=self.treatment_col,
                ps_col=self.ps_col,
                pid_col="index",
                caliper=self.caliper,
                n_controls=self.n_controls,
                strict=self.strict,
            )

        if self.effect_type in ["ATE", "ARR"]:
            warnings.warn(
                "This is strictly speaking not ATE if we used a caliper or other matching methods. But can be interpreted as such."
            )
            return compute_matching_ate(Y, matched)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
