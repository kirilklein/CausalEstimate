import numpy as np

from CausalEstimate.utils.constants import (
    INITIAL_EFFECT,
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
)


def compute_initial_effect(Y1_hat, Y0_hat, Q_star_1, Q_star_0, rr=False) -> dict:
    """
    Compute the initial effect and adjustments.

    Parameters:
    -----------
    Y1_hat: array-like
        Initial outcome prediction for treatment group
    Y0_hat: array-like
        Initial outcome prediction for control group
    Q_star_1: array-like
        Updated outcome predictions for treatment group
    Q_star_0: array-like
        Updated outcome predictions for control group
    rr: bool, optional
        If True, compute the risk ratio. If False, compute the average treatment effect.

    Returns:
    --------
    dict: A dictionary containing the initial effect and adjustments.
    """
    initial_effect_1 = Y1_hat.mean()
    initial_effect_0 = Y0_hat.mean()

    if rr:
        if initial_effect_0 == 0:
            import warnings

            warnings.warn(
                "Initial effect for untreated group is 0, risk ratio undefined",
                RuntimeWarning,
            )
            initial_effect = np.inf
        else:
            initial_effect = initial_effect_1 / initial_effect_0
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
