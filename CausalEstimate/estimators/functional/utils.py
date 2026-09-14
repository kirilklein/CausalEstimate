import numpy as np
import warnings
from typing import Literal, NamedTuple, Tuple

from CausalEstimate.utils.constants import (
    INITIAL_EFFECT,
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
)
from CausalEstimate.utils.checks import check_ps_not_exact_zero_one
from statsmodels.genmod.families import Binomial
from statsmodels.tools import sm_exceptions
from statsmodels.genmod.generalized_linear_model import GLM
from scipy.special import expit, logit

MARGINAL_EFFECTS = ("ATE", "RR")
TREATED_EFFECTS = ("ATT",)
EffectType = Literal["ATE", "RR", "ATT"]


class TargetingResult(NamedTuple):
    """
    The output of the targeting step.

    Attributes
    ----------
    Q_star_1, Q_star_0 : targeted outcome predictions under treatment and
        control, for all units.
    Yhat_star : targeted prediction at the observed treatment. Derived from
        Q_star_1/Q_star_0, so it agrees with them by construction.
    w1, w0 : the targeting weights the fluctuation was fitted with, zero
        off-arm. Pass these to the variance step rather than recomputing.
    epsilon_1, epsilon_0 : the fitted logit-scale shifts, kept for diagnostics.
    """

    Q_star_1: np.ndarray
    Q_star_0: np.ndarray
    Yhat_star: np.ndarray
    w1: np.ndarray
    w0: np.ndarray
    epsilon_1: float
    epsilon_0: float

    @property
    def H(self) -> np.ndarray:
        """
        Combined influence-curve covariate, w1 - w0.

        For ATE/RR this is A/ps - (1-A)/(1-ps): the signed clever covariate
        the variance step expects, but built from the clipped weights actually
        used in the fit rather than recomputed from ps (issue #98).
        """
        return self.w1 - self.w0


# --- Targeting weights -------------------------------------------------------


def compute_arm_weights(
    A: np.ndarray,
    ps: np.ndarray,
    effect_type: EffectType = "ATE",
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-arm targeting weights, zero off-arm:

        ATE / RR   (marginal arm means):
            w1 = A/ps                 w0 = (1-A)/(1-ps)
        ATT        (treated subpopulation):
            w1 = A/p_treated          w0 = (1-A) * ps/(p_treated (1-ps))

    Both weights are non-negative; the direction of each fluctuation is
    determined by the root of its score equation, not by a sign convention on
    the covariate.

    Parameters
    ----------
    A: Binary treatment assignment (0 or 1).
    ps: Propensity scores.
    effect_type: One of "ATE", "RR", "ATT".
    clip_percentile: Upper quantile at which to clip each arm's weights, as a
        PROPORTION in (0, 1]. 1 (default) applies no clipping; 0.99 clips the
        top 1%. This is the same convention as `compute_ipw_weights`.
    eps: Small constant added to denominators. Note this does not bound the
        weights: at ps = 0, 1/(ps + 1e-9) is 1e9. Use `clip_percentile` or
        trim upstream if that matters.
    """
    if effect_type not in MARGINAL_EFFECTS + TREATED_EFFECTS:
        raise ValueError(
            f"effect_type '{effect_type}' is not supported. "
            f"Expected one of {MARGINAL_EFFECTS + TREATED_EFFECTS}."
        )
    if not (0 < clip_percentile <= 1.0):
        raise ValueError("clip_percentile must be in the interval (0, 1.0].")

    if effect_type in MARGINAL_EFFECTS:
        w1 = A / (ps + eps)
        w0 = (1 - A) / (1 - ps + eps)
    else:
        p_treated = float(np.mean(A == 1))
        if p_treated == 0.0:
            warnings.warn(
                "No treated subjects found; targeting weights are all zero and "
                "the targeting step will be a no-op. The reported estimate is "
                "the untargeted initial estimate.",
                RuntimeWarning,
            )
            zeros = np.zeros_like(ps, dtype=float)
            return zeros, zeros
        w1 = A / p_treated
        w0 = (1 - A) * ps / (p_treated * (1 - ps) + eps)

    treated_mask = A == 1
    control_mask = ~treated_mask
    if clip_percentile < 1:
        w1 = _clip_on_arm(w1, treated_mask, clip_percentile)
        w0 = _clip_on_arm(w0, control_mask, clip_percentile)

    _warn_extreme_weights(w1, treated_mask, f"{effect_type} treated")
    _warn_extreme_weights(w0, control_mask, f"{effect_type} control")
    return w1, w0


def _clip_on_arm(
    weights: np.ndarray, arm_mask: np.ndarray, clip_percentile: float
) -> np.ndarray:
    """
    Clip an arm's weights at a quantile computed over that arm. Off-arm
    entries are zero and are left alone.

    `clip_percentile` is a PROPORTION in (0, 1], not a percentage: 0.99 clips
    the top 1%.
    """
    if arm_mask.sum() == 0:
        return weights
    threshold = np.percentile(weights[arm_mask], clip_percentile * 100)
    out = weights.copy()
    out[arm_mask] = np.clip(weights[arm_mask], a_min=None, a_max=threshold)
    return out


def _warn_extreme_weights(
    weights: np.ndarray, arm_mask: np.ndarray, label: str, threshold: float = 100
) -> None:
    """
    Warn once per arm when the targeting weights are large enough to make the
    fluctuation unstable, reporting how many units are involved.

    Only on-arm weights are inspected: off-arm entries are zero by
    construction, and a small propensity score inflates the OTHER arm's
    weight, which is where it actually matters.
    """
    if arm_mask.sum() == 0:
        return
    on_arm = weights[arm_mask]
    extreme = on_arm > threshold
    if extreme.any():
        warnings.warn(
            f"{extreme.sum()} of {arm_mask.sum()} units on the {label} arm have "
            f"targeting weights above {threshold} (max {on_arm.max():.1f}). "
            "This usually means propensity scores near 0 or 1; consider "
            "clip_percentile or trimming.",
            RuntimeWarning,
        )


# --- Fluctuation -------------------------------------------------------------


def estimate_arm_fluctuation(
    Y: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    label: str,
    max_shift: float = 30.0,
    q_eps: float = 1e-6,
) -> float:
    """
    Estimate the logit-scale shift epsilon for ONE arm.

    Each arm gets its own epsilon; `target_outcome_models` calls this twice.
    That is the difference from the single fluctuation parameter this replaced,
    which could only zero the two arms' scores jointly, not individually
    (issue #140).

    Fits the intercept-only weighted logistic fluctuation

        Y ~ expit(logit(Q) + epsilon),  weights w

    as a statsmodels GLM with `logit(Q)` as the offset. Its score equation is

        s(epsilon) = sum(w * (Y - expit(logit(Q) + epsilon))) = 0

    and `s` is strictly decreasing in epsilon, so the root is unique whenever
    one exists.

    A score that keeps its sign at +/- max_shift means the maximiser is at
    +/- infinity: the arm is separated (for instance every weighted outcome is
    0, so Q* wants to be 0). The GLM would not converge there, so the shift is
    clamped to +/- max_shift and a warning issued. expit(+/-30) is already
    within 1e-13 of 0 or 1, so the clamp costs nothing numerically and only
    prevents overflow.

    Parameters
    ----------
    Y: Observed outcomes.
    Q: Initial predictions for this arm, for all units.
    w: This arm's targeting weights, zero off-arm.
    label: Arm description used in warnings.
    max_shift: Bound on the logit-scale shift.
    q_eps: Bound applied to Q before the logit.
    """
    if not np.any(w > 0):
        warnings.warn(
            f"No units carry weight on the {label} arm; skipping its "
            "fluctuation (epsilon = 0).",
            RuntimeWarning,
        )
        return 0.0

    offset = logit(np.clip(Q, q_eps, 1 - q_eps))

    def score(epsilon: float) -> float:
        return float(np.sum(w * (Y - expit(offset + epsilon))))

    if score(max_shift) > 0:
        warnings.warn(
            f"Fluctuation for the {label} arm is unbounded above (separated "
            f"arm); clamping the logit-scale shift to +{max_shift:.0f}.",
            RuntimeWarning,
        )
        return float(max_shift)

    if score(-max_shift) < 0:
        warnings.warn(
            f"Fluctuation for the {label} arm is unbounded below (separated "
            f"arm); clamping the logit-scale shift to -{max_shift:.0f}.",
            RuntimeWarning,
        )
        return float(-max_shift)

    intercept = np.ones((Y.size, 1))
    try:
        fit = GLM(Y, intercept, family=Binomial(), offset=offset, freq_weights=w).fit(
            tol=1e-12, maxiter=100
        )
        epsilon = float(np.asarray(fit.params)[0])
    except (np.linalg.LinAlgError, sm_exceptions.PerfectSeparationError):
        epsilon = np.nan

    # Judge the fit by its own score equation rather than by `fit.converged`.
    # statsmodels compares the *absolute* deviance change to `tol`, and
    # freq_weights scale the deviance, so an arm carrying large IPW weights
    # reports converged=False at tol=1e-12 even when epsilon is exact to
    # machine precision. The score is what we actually need solved.
    solved = np.isfinite(epsilon) and abs(score(epsilon)) <= 1e-8 * max(
        float(np.sum(w)), 1.0
    )

    if not solved:
        warnings.warn(
            f"The GLM fluctuation for the {label} arm did not solve its score "
            "equation; falling back to the one-step estimate.",
            RuntimeWarning,
        )
        return _one_step_fluctuation(Y, offset, w, label, max_shift)

    return epsilon


def _one_step_fluctuation(
    Y: np.ndarray,
    offset: np.ndarray,
    w: np.ndarray,
    label: str,
    max_shift: float,
) -> float:
    """
    Non-iterative fallback for epsilon, used when the GLM fit fails.

    This is the first Newton-Raphson step on the weighted score equation
    s(epsilon) = sum(w * (Y - expit(offset + epsilon))), started from
    epsilon = 0:

        epsilon_1-step = s(0) / I(0)

    with observed information I(0) = sum(w * Q * (1 - Q)), Q = expit(offset).
    The covariate is the intercept, so the weights enter both terms linearly.

    The result is clamped to +/- max_shift for the same reason the GLM path is:
    a near-zero information can otherwise send the step off to a shift that
    overflows expit.
    """
    Q = expit(offset)
    score = float(np.sum(w * (Y - Q)))
    information = float(np.sum(w * Q * (1 - Q)))

    if not np.isfinite(score) or not np.isfinite(information) or information == 0:
        warnings.warn(
            f"Non-finite score or zero information in the one-step estimate "
            f"for the {label} arm; falling back to epsilon = 0, i.e. the "
            "untargeted fit for this arm.",
            RuntimeWarning,
        )
        return 0.0
    return float(np.clip(score / information, -max_shift, max_shift))


def target_outcome_models(
    A: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    Y1_hat: np.ndarray,
    Y0_hat: np.ndarray,
    effect_type: EffectType = "ATE",
    clip_percentile: float = 1,
    eps: float = 1e-9,
    q_eps: float = 1e-6,
    max_shift: float = 30.0,
) -> TargetingResult:
    """
    Run the full targeting step for any of the four estimands.

    This is the single entry point every TMLE estimator should use. The
    estimands differ only in their weights (see `compute_arm_weights`) and in
    how the targeted arm means are combined afterwards.
    """
    w1, w0 = compute_arm_weights(A, ps, effect_type, clip_percentile, eps)

    epsilon_1 = estimate_arm_fluctuation(
        Y, Y1_hat, w1, f"{effect_type} treated", max_shift=max_shift, q_eps=q_eps
    )
    epsilon_0 = estimate_arm_fluctuation(
        Y, Y0_hat, w0, f"{effect_type} control", max_shift=max_shift, q_eps=q_eps
    )

    Q_star_1 = expit(logit(np.clip(Y1_hat, q_eps, 1 - q_eps)) + epsilon_1)
    Q_star_0 = expit(logit(np.clip(Y0_hat, q_eps, 1 - q_eps)) + epsilon_0)
    Yhat_star = np.where(A == 1, Q_star_1, Q_star_0)

    return TargetingResult(
        Q_star_1=Q_star_1,
        Q_star_0=Q_star_0,
        Yhat_star=Yhat_star,
        w1=w1,
        w0=w0,
        epsilon_1=epsilon_1,
        epsilon_0=epsilon_0,
    )


# --- Diagnostics -------------------------------------------------------------


def check_score_equations(
    result: TargetingResult, Y: np.ndarray
) -> Tuple[float, float]:
    """
    Verify that both arm score equations are solved, returning the two scores.

    Warns rather than asserts: a bare assert vanishes under `python -O`, and a
    clamped (separated) arm legitimately leaves a non-zero score. The
    tolerance is 1% of each score's own standard error, so it scales with the
    weights instead of assuming they are O(1).
    """
    return (
        _check_one_score(result.w1, Y, result.Q_star_1, "treated"),
        _check_one_score(result.w0, Y, result.Q_star_0, "control"),
    )


def _check_one_score(
    w: np.ndarray, Y: np.ndarray, Q_star: np.ndarray, label: str
) -> float:
    contributions = w * (Y - Q_star)
    score = float(np.mean(contributions))
    n = contributions.size
    se = float(np.std(contributions, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tol = max(0.01 * se, 1e-8)

    if not np.isfinite(score) or abs(score) > tol:
        warnings.warn(
            f"Score equation for the {label} arm is not solved: mean = "
            f"{score:.3e} (tolerance {tol:.3e}). The estimate may not be "
            "doubly robust; this normally means the arm was separated and its "
            "fluctuation was clamped.",
            RuntimeWarning,
        )
    return score


# --- Effect helpers ----------------------------------------------------------


def safe_ratio(
    numerator: float, denominator: float, label: str = "Risk ratio"
) -> float:
    """
    Numerator/denominator with one consistent set of guards, shared by the
    initial and targeted ratio computations so they cannot disagree.

    Returns np.inf for a (near-)zero denominator or a ratio above 1e5, np.nan
    if either input is already non-finite. Emits at most one warning.
    """
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        warnings.warn(
            f"{label} has non-finite inputs; returning np.nan.", RuntimeWarning
        )
        return np.nan
    if np.isclose(denominator, 0.0, atol=1e-8):
        warnings.warn(
            f"{label} denominator is 0 or nearly 0; returning np.inf.",
            RuntimeWarning,
        )
        return np.inf

    ratio = numerator / denominator
    if not np.isfinite(ratio) or abs(ratio) > 1e5:
        warnings.warn(
            f"{label} is unrealistically large ({ratio:.2e}); returning np.inf.",
            RuntimeWarning,
        )
        return np.inf
    return float(ratio)


def compute_initial_effect(
    Y1_hat: np.ndarray,
    Y0_hat: np.ndarray,
    Q_star_1: np.ndarray,
    Q_star_0: np.ndarray,
    rr: bool = False,
) -> dict:
    """
    Untargeted effect plus the size of the targeting adjustment in each arm.

    Parameters
    ----------
    Y1_hat, Y0_hat : initial outcome predictions under treatment / control.
    Q_star_1, Q_star_0 : targeted predictions under treatment / control.
    rr : if True the initial effect is the ratio of arm means (guarded by
        `safe_ratio`, so it may be np.inf or np.nan); otherwise the difference.

    Returns
    -------
    dict with keys INITIAL_EFFECT, INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated, ADJUSTMENT_treated, ADJUSTMENT_untreated. The
    adjustments are mean(Q* - Q) per arm: near-zero means the targeting step
    barely moved the initial fit.
    """
    initial_effect_1 = float(Y1_hat.mean())
    initial_effect_0 = float(Y0_hat.mean())

    if rr:
        initial_effect = safe_ratio(
            initial_effect_1, initial_effect_0, label="Initial risk ratio"
        )
    else:
        initial_effect = initial_effect_1 - initial_effect_0

    adjustment_1 = (Q_star_1 - Y1_hat).mean()
    adjustment_0 = (Q_star_0 - Y0_hat).mean()
    return {
        INITIAL_EFFECT: initial_effect,
        INITIAL_EFFECT_treated: initial_effect_1,
        INITIAL_EFFECT_untreated: initial_effect_0,
        ADJUSTMENT_treated: adjustment_1,
        ADJUSTMENT_untreated: adjustment_0,
    }


# --- Centralized Weight Calculation Functions --------------------------------


def compute_ipw_weights(
    A: np.ndarray,
    ps: np.ndarray,
    weight_type: Literal["ATE", "ATT"] = "ATE",
    clip_percentile: float = 1,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Inverse Propensity Score weights with optional per-arm clipping.

    Formulas:
    - ATE: w = A/ps + (1-A)/(1-ps)
    - ATT: w = A + (1-A) * ps/(1-ps)

    Note these are the IPW estimator's weights, which differ from the TMLE
    targeting weights of `compute_arm_weights`: the ATT variants are
    normalised differently (1 vs 1/p_treated on the treated arm), and here the
    two arms are summed into a single vector rather than kept separate.

    Args:
        A: Binary treatment assignment vector (1 for treated, 0 for control).
        ps: Propensity score vector.
        weight_type: "ATE" or "ATT".
        clip_percentile: Upper quantile at which to clip, as a PROPORTION in
            (0, 1]. 1.0 (default) applies no clipping; 0.99 clips the top 1%.
        eps: Small constant added to denominators.

    Returns:
        An array of IPW weights.

    Raises:
        ValueError: on an invalid `weight_type`, mismatched shapes, or a
            `clip_percentile` outside (0, 1].
    """
    if weight_type not in ("ATE", "ATT"):
        raise ValueError("weight_type must be 'ATE' or 'ATT'")
    if not (0 < clip_percentile <= 1.0):
        raise ValueError("clip_percentile must be in the interval (0, 1.0].")
    if A.shape != ps.shape:
        raise ValueError("A and ps must have the same shape.")
    check_ps_not_exact_zero_one(ps)

    if weight_type == "ATE":
        weights = A / (ps + eps) + (1 - A) / (1 - ps + eps)
    else:  # ATT: treated units have weight 1.
        weights = A + (1 - A) * ps / (1 - ps + eps)

    if clip_percentile < 1.0:
        q = clip_percentile * 100
        treated_mask = A == 1
        for mask in (treated_mask, ~treated_mask):
            if np.any(mask):
                threshold = np.percentile(weights[mask], q)
                weights[mask] = np.clip(weights[mask], a_min=None, a_max=threshold)

    return weights
