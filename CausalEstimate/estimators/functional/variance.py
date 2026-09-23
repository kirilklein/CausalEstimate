import numpy as np

from CausalEstimate.estimators.functional.utils import RATIO_DENOM_ATOL
from CausalEstimate.utils.constants import CI95_LOWER, CI95_UPPER, STD_ERR

RATIO_EFFECTS = ("RR", "RRT")


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

    The targeting step solves mean(w * (Y - Q_star)) = 0 for each arm by
    construction, so the plug-in arm means are already correctly centred and
    the canonical influence function carries no Hajek normaliser -- hence
    normalize=False in the RR branch below. AIPW and IPW, whose point
    estimates use self-normalised weights, go through compute_ci_aipw() and
    compute_ci_ipw() instead.

    The nuisance models are treated as fixed; the bootstrap remains the option
    that accounts for their estimation. Any clipping applied to the targeting
    weights is also treated as fixed, including data-adaptive clipping through
    clip_percentile. In settings where clip_percentile < 1, bootstrap is recommended.
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
        mu_1 = float(Q_star_1.mean())
        mu_0 = float(Q_star_0.mean())
        ic_mu1 = _compute_ic_mu(Y, w1, Q_star_1, mu_1, normalize=False, eps=eps)
        ic_mu0 = _compute_ic_mu(Y, w0, Q_star_0, mu_0, normalize=False, eps=eps)
        ic = _compute_ic_log_ratio(ic_mu1, ic_mu0, mu_1, mu_0, eps)
    else:
        raise ValueError(
            f"CI calculation for effect type '{effect_type}' is not supported."
        )

    return _summarise_ic(effect_type, psi, ic)


def compute_ci_aipw(
    effect_type: str,
    psi: float,
    Y: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    Q_1: np.ndarray,
    Q_0: np.ndarray,
    mu_1: float,
    mu_0: float,
    eps: float = 1e-9,
) -> dict:
    """
    Standard error and 95% CI for the AIPW estimators via the influence curve.

    W must be the same weight vector used for the point estimate (including any
    clipping), and mu_1/mu_0 the same point estimates -- NOT mean(Q_1) and
    mean(Q_0), since the augmentation term shifts them.

    The point estimate is a ratio, mu = Qbar + mean(w (Y - Q)) / mean(w), so
    the influence curve is the ratio's -- both numerator and denominator are
    differentiated. See _compute_ic_mu for the resulting r term. Under a
    correctly specified outcome model r -> 0 and this agrees with the plain
    (and with the un-normalised) form; under misspecification it does not, and
    the ratio curve is the one that matches the estimate actually reported.

    Q_1 is unused for ATT, where mu_1 is the observed treated mean.

    The nuisance models are treated as fixed; the bootstrap remains the option
    that accounts for their estimation. Any clipping applied to W is also
    treated as fixed, including data-adaptive clipping through clip_percentile.
    In settings where clip_percentile < 1, bootstrap is recommended.
    """
    if len(Y) == 0:
        return {STD_ERR: np.nan, CI95_LOWER: np.nan, CI95_UPPER: np.nan}

    w1, w0 = A * W, (1 - A) * W

    if effect_type in ["ATE", "ARR"]:
        ic_mu1 = _compute_ic_mu(Y, w1, Q_1, mu_1, eps=eps)
        ic_mu0 = _compute_ic_mu(Y, w0, Q_0, mu_0, eps=eps)
    elif effect_type == "ATT":
        p_treated = np.mean(A)
        if np.isclose(p_treated, 0.0, atol=eps):
            return {STD_ERR: np.nan, CI95_LOWER: np.nan, CI95_UPPER: np.nan}
        A_over_p = A / p_treated
        # mu_1 is the raw treated mean, so its IC is the treated-restricted
        # deviation; mu_0 carries the augmentation, centred on A/p.
        ic_mu1 = A_over_p * (Y - mu_1)
        ic_mu0 = _compute_ic_mu(Y, w0, Q_0, mu_0, A_over_p=A_over_p, eps=eps)
    else:
        raise ValueError(
            f"CI calculation for effect type '{effect_type}' is not supported."
        )

    ic = ic_mu1 - ic_mu0

    return _summarise_ic(effect_type, psi, ic)


def compute_ci_ipw(
    effect_type: str,
    psi: float,
    Y: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    mu_1: float,
    mu_0: float,
    eps: float = 1e-9,
) -> dict:
    """
    Standard error and 95% CI for the Hajek IPW estimators via the influence
    curve. Supports ATE, ATT, RR and RRT -- the estimand enters only through W
    (ATE weights for ATE/RR, ATT weights for ATT/RRT).

    W must be the same weight vector used for the point estimate (including any
    clipping) and mu_1/mu_0 the same weighted means, so the influence curve is
    exactly mean-zero and the SE stays consistent with the estimate.

    IPW is the degenerate case of the AIPW decomposition with Q set to the
    constant mu: the plug-in term Q - mu vanishes and each arm contributes
    w (Y - mu) / mean(w). Normalisation is not optional here -- unlike AIPW,
    mean(w (Y - mu)) does not tend to zero, so dropping it gives the
    Horvitz-Thompson influence curve and an inflated SE.

    The propensity score is treated as fixed; the bootstrap remains the option
    that accounts for nuisance estimation. Any clipping applied to W is also
    treated as fixed, including data-adaptive clipping through clip_percentile.
    In settings where clip_percentile < 1, bootstrap is recommended.
    """
    if len(Y) == 0:
        return {STD_ERR: np.nan, CI95_LOWER: np.nan, CI95_UPPER: np.nan}

    if effect_type not in ["ATE", "ATT", "ARR", "RR", "RRT"]:
        raise ValueError(
            f"CI calculation for effect type '{effect_type}' is not supported."
        )

    w1, w0 = A * W, (1 - A) * W
    ic_mu1 = _compute_ic_mu(Y, w1, np.full(Y.shape, mu_1), mu_1, eps=eps)
    ic_mu0 = _compute_ic_mu(Y, w0, np.full(Y.shape, mu_0), mu_0, eps=eps)

    if effect_type in RATIO_EFFECTS:
        ic = _compute_ic_log_ratio(ic_mu1, ic_mu0, mu_1, mu_0, eps)
    else:
        ic = ic_mu1 - ic_mu0

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
    normalize: bool = True,
    A_over_p: np.ndarray = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Influence curve for a single arm mean,

        IC_i = w_i (Y_i - Q_i - r) / d + c_i (Q_i - Qbar)

    with d = mean(w) when normalize else 1, and c = 1 for unconditional means
    or A/P(A=1) for treated-restricted ones (ATT).

    w is that arm's own non-negative weight, zero off-arm.

    Setting Q to the constant mu recovers the plain Hajek IPW curve; passing
    the outcome regressions gives AIPW, and the targeted Q_star gives TMLE
    (with normalize=False, since targeting already solves the score equation).

    The self-normalised arm mean is a ratio, mu = Qbar + mean(w (Y - Q)) / d,
    so differentiating it holds the denominator responsible too: the quotient
    rule leaves r = mean(w (Y - Q)) / d inside the residual, not outside it.
    Dropping it -- i.e. subtracting the constant r rather than r w / d -- costs
    a term r (w / d - 1), which vanishes only when r = 0, that is when the
    outcome model is correctly specified. Under misspecification it makes
    the SE inconsistent, and weight clipping does not by itself make r nonzero.

    Qbar is recovered as mu - r rather than passed in, so the curve is pinned
    to the caller's own point estimate; with normalize=False there is no ratio
    and r is identically zero.

    Mean-zero by construction -- but note that BOTH the correct curve and the
    r-outside-the-residual one are mean-zero, so that property alone does not
    catch the denominator contribution.
    """
    if np.isnan(mu):
        return np.full(Y.shape, np.nan, dtype=float)
    denom = w.mean() if normalize else 1.0
    if np.isclose(denom, 0.0, atol=eps):
        return np.full(Y.shape, np.nan, dtype=float)
    # Hajek denominator contribution: zero unless the estimate is a ratio.
    r = (w * (Y - Q)).mean() / denom if normalize else 0.0
    Q_bar = mu - r
    centre = (Q - Q_bar) if A_over_p is None else A_over_p * (Q - Q_bar)
    return w * (Y - Q - r) / denom + centre


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

    if effect_type in RATIO_EFFECTS:
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
