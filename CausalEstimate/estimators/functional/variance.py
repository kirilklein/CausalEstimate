import numpy as np

from CausalEstimate.estimators.functional.utils import RATIO_DENOM_ATOL
from CausalEstimate.utils.constants import CI95_LOWER, CI95_UPPER, STD_ERR


def compute_ci(
    effect_type: str,
    psi: float,
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    Yhat_star: np.ndarray,
    H: np.ndarray = None,
    w1: np.ndarray = None,
    w0: np.ndarray = None,
    eps: float = 1e-9,
) -> dict:
    """
    Standard error and 95% confidence interval for the TMLE estimators, from
    the influence curve.

    Difference effects combine the arm weights as H = w1 − w0;
    RR uses the arm weights separately.

    w1 and w0 are the non-negative, off-arm-zero weights carried by
    `TargetingResult`; pass them through unchanged. Each arm mean's influence
    curve uses that arm's own weight with a POSITIVE sign -- it is only the
    difference covariate H = w1 - w0 that carries the control-arm minus sign.
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
        if w1 is None or w0 is None:
            raise ValueError(
                "effect_type 'RR' requires the arm-wise targeting weights w1 "
                "and w0 from the targeting step."
            )
        mu_1 = Q_star_1.mean()
        mu_0 = Q_star_0.mean()
        ic_mu1 = _compute_ic_mu(Y, w1, Q_star_1, mu_1)
        ic_mu0 = _compute_ic_mu(Y, w0, Q_star_0, mu_0)
        ic = _compute_ic_log_ratio(ic_mu1, ic_mu0, mu_1, mu_0, eps)
    else:
        raise ValueError(
            f"CI calculation for effect type '{effect_type}' is not supported."
        )

    return _summarise_ic(effect_type, psi, ic)


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


def _compute_ic_mu(
    Y: np.ndarray,
    w: np.ndarray,
    Q: np.ndarray,
    mu: float,
) -> np.ndarray:
    """
    Influence curve for a single targeted arm mean,

        IC_i = w_i (Y_i - Q_i) + (Q_i - mu)

    w is that arm's own non-negative weight, zero off-arm, so no masking by A
    is needed here and the residual term keeps a positive sign.

    The weighted residual is not divided by mean(w): the targeting step has
    already solved this arm's score equation, so the term is centred already.

    Mean-zero by construction, which is the cheapest available regression test.
    """
    if np.isnan(mu):
        return np.full(Y.shape, np.nan, dtype=float)
    return w * (Y - Q) + (Q - mu)


def _compute_ic_log_ratio(
    ic_mu1: np.ndarray,
    ic_mu0: np.ndarray,
    mu_1: float,
    mu_0: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Delta-method influence curve for log(mu_1 / mu_0).

    Non-positive arm means give NaN: the log scale is undefined there, and a
    negative weighted mean (possible with extreme weights) would otherwise
    propagate silently.
    """
    if np.isnan(mu_1) or np.isnan(mu_0):
        return np.full(ic_mu1.shape, np.nan, dtype=float)
    if mu_1 <= eps or mu_0 <= eps or np.isclose(mu_0, 0.0, atol=RATIO_DENOM_ATOL):
        return np.full(ic_mu1.shape, np.nan, dtype=float)
    return ic_mu1 / mu_1 - ic_mu0 / mu_0


def _summarise_ic(effect_type: str, psi: float, ic: np.ndarray) -> dict:
    """Standard error and 95% CI from a mean-zero influence curve."""
    if not np.isfinite(psi) or np.any(np.isnan(ic)):
        return {STD_ERR: np.nan, CI95_LOWER: np.nan, CI95_UPPER: np.nan}

    n = len(ic)
    var_ic = np.var(ic, ddof=1)  # Use ddof=1 for sample variance
    std_err_ic = np.sqrt(var_ic / n)

    if effect_type == "RR":
        # The IC is on the log scale, so the SE is too and the CI is
        # exponentiated. Keeps CI95 == exp(log(psi) +/- 1.96 * STD_ERR).
        log_psi = np.log(psi)
        return {
            STD_ERR: std_err_ic,
            CI95_LOWER: np.exp(log_psi - 1.96 * std_err_ic),
            CI95_UPPER: np.exp(log_psi + 1.96 * std_err_ic),
        }
    return {
        STD_ERR: std_err_ic,
        CI95_LOWER: psi - 1.96 * std_err_ic,
        CI95_UPPER: psi + 1.96 * std_err_ic,
    }
