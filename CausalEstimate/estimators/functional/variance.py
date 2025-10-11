import numpy as np

from CausalEstimate.utils.constants import CI95_LOWER, CI95_UPPER, STD_ERR


def compute_ci(
    effect_type: str,
    psi: float,
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    ps: np.ndarray,
    Yhat_star: np.ndarray,
    H: np.ndarray = None,
) -> dict:
    """
    Compute the standard deviation and 95% confidence interval using the influence curve.
    """
    n = len(Y)
    if n == 0:
        return {STD_ERR: np.nan, CI95_LOWER: np.nan, CI95_UPPER: np.nan}

    # Select the appropriate influence curve based on the effect type
    if effect_type in ["ATE", "ARR"]:
        ic = _compute_ic_ate(psi, Q_star_1, Q_star_0, Y, A, Yhat_star, H)
    elif effect_type == "ATT":
        p_treated = np.mean(A)
        ic = _compute_ic_att(psi, Q_star_1, Q_star_0, Y, A, Yhat_star, H, p_treated)
    elif effect_type == "RR":
        ic = _compute_ic_rr(Q_star_1, Q_star_0, Y, A, ps)
    else:
        raise ValueError(
            f"CI calculation for effect type '{effect_type}' is not supported."
        )

    if np.any(np.isnan(ic)):
        return {STD_ERR: np.nan, CI95_LOWER: np.nan, CI95_UPPER: np.nan}

    # Compute variance and standard error
    var_ic = np.var(ic, ddof=1)  # Use ddof=1 for sample variance
    std_err = np.sqrt(var_ic / n)

    # Compute confidence interval
    if effect_type == "RR":
        # For RR, CIs are calculated on the log scale and then exponentiated
        log_psi = np.log(psi)
        ci_lower = np.exp(log_psi - 1.96 * std_err)
        ci_upper = np.exp(log_psi + 1.96 * std_err)
    else:  # ATE, ATT, ARR
        ci_lower = psi - 1.96 * std_err
        ci_upper = psi + 1.96 * std_err

    return {STD_ERR: std_err, CI95_LOWER: ci_lower, CI95_UPPER: ci_upper}


def _compute_ic_ate(
    psi: float,
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    Yhat_star: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Influence curve for ATE."""
    return H * (Y - Yhat_star) + (Q_star_1 - Q_star_0) - psi


def _compute_ic_att(
    psi: float,
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    Yhat_star: np.ndarray,
    H: np.ndarray,
    p_treated: float,
) -> np.ndarray:
    """Influence curve for ATT."""
    if np.isclose(p_treated, 0.0, atol=1e-12):
        return np.full(Y.shape, np.nan, dtype=float)
    ic = H * (Y - Yhat_star) + (A / p_treated) * (Q_star_1 - Q_star_0 - psi)
    return ic


def _compute_ic_rr(
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    ps: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Influence curve for log(Risk Ratio)."""
    mu1_star = np.mean(Q_star_1)
    mu0_star = np.mean(Q_star_0)

    if np.isclose(mu0_star, 0.0, atol=eps) or np.isclose(mu1_star, 0.0, atol=eps):
        return np.full(Y.shape, np.nan, dtype=float)

    # IC for mu1
    ic_mu1 = (A / (ps + eps)) * (Y - Q_star_1) + Q_star_1 - mu1_star
    # IC for mu0
    ic_mu0 = ((1 - A) / (1 - ps + eps)) * (Y - Q_star_0) + Q_star_0 - mu0_star

    ic_log_rr = (1 / mu1_star) * ic_mu1 - (1 / mu0_star) * ic_mu0
    return ic_log_rr
