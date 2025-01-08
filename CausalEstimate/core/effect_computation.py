import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from CausalEstimate.core.bootstrap import generate_bootstrap_samples
from CausalEstimate.core.logging import log_initial_stats, log_sample_stats
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def compute_effects(
    estimators: List,
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    ps_col: str,
    bootstrap: bool,
    n_bootstraps: int,
    method_args: Dict,
    apply_common_support: bool,
    common_support_threshold: float,
    **kwargs,
) -> Dict:
    """
    Compute causal effects using specified estimators.

    Args:
        estimators: List of estimator objects
        df: Input DataFrame
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        ps_col: Name of propensity score column
        bootstrap: Whether to use bootstrapping
        n_bootstraps: Number of bootstrap iterations
        method_args: Additional arguments for estimators
        apply_common_support: Whether to apply common support
        common_support_threshold: Threshold for common support
        **kwargs: Additional keyword arguments

    Returns:
        Dictionary of computed effects
    """
    log_initial_stats(df, treatment_col, outcome_col, ps_col)

    if bootstrap:
        return compute_bootstrap_effects(
            estimators=estimators,
            df=df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            n_bootstraps=n_bootstraps,
            method_args=method_args,
            apply_common_support=apply_common_support,
            common_support_threshold=common_support_threshold,
            **kwargs,
        )
    else:
        return compute_single_effect(
            estimators=estimators,
            df=df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            method_args=method_args,
            apply_common_support=apply_common_support,
            common_support_threshold=common_support_threshold,
            **kwargs,
        )


def compute_bootstrap_effects(
    estimators: List,
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    ps_col: str,
    n_bootstraps: int,
    method_args: Dict,
    apply_common_support: bool,
    common_support_threshold: float,
    **kwargs,
):
    """
    Compute effects using bootstrap sampling.
    """
    bootstrap_samples = generate_bootstrap_samples(df, n_bootstraps)
    results = {type(estimator).__name__: [] for estimator in estimators}
    logger.info(f"Applying common support {apply_common_support}")
    for sample in tqdm(bootstrap_samples, desc="Computing effects"):
        sample = (
            filter_common_support(
                sample,
                ps_col=ps_col,
                treatment_col=treatment_col,
                threshold=common_support_threshold,
            ).reset_index(drop=True)
            if apply_common_support
            else sample
        )

        compute_effects_for_sample(
            estimators=estimators,
            sample=sample,
            results=results,
            method_args=method_args,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            **kwargs,
        )

    return process_bootstrap_results(results, n_bootstraps)


def compute_single_effect(
    estimators: List,
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    ps_col: str,
    method_args: Dict,
    apply_common_support: bool,
    common_support_threshold: float,
    **kwargs,
):
    """
    Compute effects for a single sample.
    """
    df = (
        filter_common_support(
            df,
            ps_col=ps_col,
            treatment_col=treatment_col,
            threshold=common_support_threshold,
        ).reset_index(drop=True)
        if apply_common_support
        else df
    )

    log_sample_stats(df, treatment_col, outcome_col, ps_col)

    results = {type(estimator).__name__: [] for estimator in estimators}
    compute_effects_for_sample(
        estimators=estimators,
        sample=df,
        results=results,
        method_args=method_args,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        ps_col=ps_col,
        **kwargs,
    )

    return process_single_results(results)


def compute_effects_for_sample(
    estimators: List,
    sample: pd.DataFrame,
    results: Dict,
    method_args: Dict,
    treatment_col: str,
    outcome_col: str,
    ps_col: str,
    **kwargs,
) -> Dict[str, float]:
    """
    Compute effects for each estimator on a given sample.
    """
    method_args = method_args or {}
    for estimator in estimators:
        method_name = type(estimator).__name__
        estimator_specific_args = method_args.get(method_name, {})
        effect = estimator.compute_effect(
            df=sample,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            **estimator_specific_args,
            **kwargs,
        )
        results[method_name].append(effect)


def process_bootstrap_results(
    results: Dict[str, List[float]], n_bootstraps: int
) -> Dict[str, Dict]:
    """
    Process results from bootstrap sampling.
    """
    return {
        method_name: {
            "effect": np.mean(effects),
            "std_err": np.std(effects),
            "bootstrap": True,
            "n_bootstraps": n_bootstraps,
        }
        for method_name, effects in results.items()
    }


def process_single_results(results: Dict[str, float]) -> Dict[str, Dict]:
    """
    Process results from a single sample.
    """
    return {
        method_name: {
            "effect": effects[0],
            "std_err": None,
            "bootstrap": False,
            "n_bootstraps": 0,
        }
        for method_name, effects in results.items()
    }
