from typing import Literal, Optional

import numpy as np


def _evalue_from_rr(rr: float) -> float:
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def _to_rr(value: float, scale: str, baseline_risk: Optional[float]) -> float:
    if scale == "RD":
        value = (baseline_risk + value) / baseline_risk
    if value <= 0:
        raise ValueError(
            "Estimate and CI bounds must be positive on the risk-ratio scale."
        )
    return float(value)


def compute_evalue(
    estimate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    scale: Literal["RR", "RD"] = "RR",
    baseline_risk: Optional[float] = None,
) -> dict:
    """
    E-value for unmeasured confounding (VanderWeele & Ding, 2017).

    The E-value is the minimum strength of association, on the risk-ratio scale,
    that an unmeasured confounder would need with both treatment and outcome
    (conditional on measured covariates) to fully explain away the observed
    effect. For RR >= 1, E = RR + sqrt(RR * (RR - 1)); protective effects use
    1 / RR. The CI E-value applies the same formula to the confidence bound
    closest to the null and is 1 when the interval contains the null.

    Args:
        estimate: Effect estimate on the given scale.
        ci_lower: Lower confidence bound; pass together with ci_upper.
        ci_upper: Upper confidence bound.
        scale: "RR" for risk ratios (RR, RRT effect types), or "RD" for risk
            differences (ATE, ARR, ATT, ATC on a binary outcome). RD values are
            converted to risk ratios via (baseline_risk + RD) / baseline_risk.
        baseline_risk: Outcome risk in the untreated (e.g. the estimator's
            `effect_0`). Required when scale="RD".

    Returns:
        dict with keys "evalue" and "evalue_ci" (None if no CI was given).
    """
    if scale not in {"RR", "RD"}:
        raise ValueError("scale must be 'RR' or 'RD'.")
    if (ci_lower is None) != (ci_upper is None):
        raise ValueError("ci_lower and ci_upper must be given together.")
    if scale == "RD":
        if baseline_risk is None:
            raise ValueError("baseline_risk is required when scale='RD'.")
        if not 0 < baseline_risk <= 1:
            raise ValueError("baseline_risk must be in (0, 1].")
    if ci_lower is not None and ci_lower > ci_upper:
        raise ValueError("ci_lower must not exceed ci_upper.")

    rr = _to_rr(estimate, scale, baseline_risk)
    result = {"evalue": _evalue_from_rr(rr), "evalue_ci": None}
    if ci_lower is None:
        return result

    lo = _to_rr(ci_lower, scale, baseline_risk)
    hi = _to_rr(ci_upper, scale, baseline_risk)
    if lo <= 1 <= hi:
        result["evalue_ci"] = 1.0
    else:
        result["evalue_ci"] = _evalue_from_rr(lo if rr > 1 else hi)
    return result
