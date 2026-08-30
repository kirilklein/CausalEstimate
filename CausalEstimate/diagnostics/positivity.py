import pandas as pd

from CausalEstimate.filter.propensity import get_common_support_range
from CausalEstimate.stats.stats import compute_propensity_score_stats
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL
from CausalEstimate.utils.utils import get_treated_ps, get_untreated_ps


def compute_positivity_metrics(
    df: pd.DataFrame,
    ps_col: str = PS_COL,
    treatment_col: str = TREATMENT_COL,
    eps: float = 0.01,
    common_support_threshold: float = 0.05,
) -> dict:
    """
    Compute positivity/overlap diagnostics for the propensity score.

    Parameters:
    -----------
    df : Input DataFrame with treatment and propensity score columns.
    ps_col : Name of the propensity score column.
    treatment_col : Name of the treatment status column (1 treated, 0 control).
    eps : Propensity scores outside [eps, 1 - eps] are counted as extreme.
    common_support_threshold : Quantile threshold passed to get_common_support_range.

    Returns:
    --------
    dict with sample sizes, shares of extreme propensity scores (overall and
    per arm), the common support range and shares outside it, the KS test
    comparing the arms' propensity score distributions, and a
    flag_positivity_violation bool (True if any extreme scores are present).
    """
    treated_ps = get_treated_ps(df, treatment_col, ps_col)
    control_ps = get_untreated_ps(df, treatment_col, ps_col)
    if len(treated_ps) == 0 or len(control_ps) == 0:
        raise ValueError(
            "Both treated and control groups must be non-empty "
            f"(n_treated={len(treated_ps)}, n_control={len(control_ps)})."
        )

    ps = df[ps_col]
    support_low, support_high = get_common_support_range(
        df, treatment_col, ps_col, common_support_threshold
    )
    ks_stats = compute_propensity_score_stats(df, ps_col, treatment_col)

    def prop_extreme(s: pd.Series) -> float:
        return float(((s < eps) | (s > 1 - eps)).mean())

    def prop_outside(s: pd.Series) -> float:
        return float(((s < support_low) | (s > support_high)).mean())

    prop_ps_extreme = prop_extreme(ps)
    return {
        "n_total": int(len(df)),
        "n_treated": int(len(treated_ps)),
        "n_control": int(len(control_ps)),
        "prop_ps_below_eps": float((ps < eps).mean()),
        "prop_ps_above_1_minus_eps": float((ps > 1 - eps).mean()),
        "prop_ps_extreme": prop_ps_extreme,
        "prop_treated_ps_extreme": prop_extreme(treated_ps),
        "prop_control_ps_extreme": prop_extreme(control_ps),
        "common_support_low": float(support_low),
        "common_support_high": float(support_high),
        "prop_outside_support": prop_outside(ps),
        "prop_treated_outside_support": prop_outside(treated_ps),
        "prop_control_outside_support": prop_outside(control_ps),
        "ks_statistic": float(ks_stats["ks_statistic"]),
        "ks_p_value": float(ks_stats["p_value"]),
        "flag_positivity_violation": bool(prop_ps_extreme > 0),
    }
