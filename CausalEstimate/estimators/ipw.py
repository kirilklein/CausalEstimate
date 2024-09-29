from CausalEstimate.estimators.functional.ipw import (
    compute_ipw_ate,
    compute_ipw_ate_stabilized,
    compute_ipw_att,
    compute_ipw_risk_ratio_treated,
    compute_ipw_risk_ratio,
)


class IPW:
    def __init__(self, effect_type="ATE", **kwargs):
        self.effect_type = effect_type
        self.kwargs = kwargs

    def compute_effect(self, df, treatment_col, outcome_col, ps_col) -> float:
        """
        Compute the effect using the functional IPW.
        Available effect types: ATE, ATT, RR, RRT
        """

        A = df[treatment_col]
        Y = df[outcome_col]
        ps = df[ps_col]

        if self.effect_type == "ATE":
            if self.kwargs.get("stabilized", False):
                return compute_ipw_ate(A, Y, ps)
            else:
                return compute_ipw_ate_stabilized(A, Y, ps)
        elif self.effect_type == "ATT":
            return compute_ipw_att(A, Y, ps)
        elif self.effect_type == "RR":
            return compute_ipw_risk_ratio(A, Y, ps)
        elif self.effect_type == "RRT":
            return compute_ipw_risk_ratio_treated(A, Y, ps)
        else:
            raise ValueError(f"Effect type '{self.effect_type}' is not supported.")
