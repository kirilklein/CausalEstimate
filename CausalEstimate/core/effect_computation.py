from typing import Dict, List

import numpy as np
import pandas as pd

from CausalEstimate.core.bootstrap import generate_bootstrap_samples
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.stats.stats import (
    compute_treatment_outcome_table,
    compute_propensity_score_stats,
)
import logging

logging = logging.getLogger(__name__)


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

    initial_table = compute_treatment_outcome_table(df, treatment_col, outcome_col)
    logging.info(f"Initial patient numbers:\n{initial_table}")

    ps_stats = compute_propensity_score_stats(df, ps_col, treatment_col)
    logging.info(f"Initial propensity score stats:\n{ps_stats}")

    if bootstrap:
        logging.info("Bootstrapping")
        return compute_bootstrap_effects(
            estimators,
            df,
            treatment_col,
            outcome_col,
            ps_col,
            n_bootstraps,
            method_args,
            apply_common_support,
            common_support_threshold,
            **kwargs,
        )
    else:
        return compute_single_effect(
            estimators,
            df,
            treatment_col,
            outcome_col,
            ps_col,
            method_args,
            apply_common_support,
            common_support_threshold,
            **kwargs,
        )


def compute_bootstrap_effects(
    estimators,
    df,
    treatment_col,
    outcome_col,
    ps_col,
    n_bootstraps,
    method_args,
    apply_common_support,
    common_support_threshold,
    **kwargs,
):
    bootstrap_samples = generate_bootstrap_samples(df, n_bootstraps)
    results = {type(estimator).__name__: [] for estimator in estimators}

    for i, sample in enumerate(bootstrap_samples):
        logging.info(f"Processing bootstrap sample {i+1} of {n_bootstraps}")
        if apply_common_support:
            logging.info("Filtering common support")
            sample = filter_common_support(
                sample,
                ps_col=ps_col,
                treatment_col=treatment_col,
                threshold=common_support_threshold,
            )

        sample_table = compute_treatment_outcome_table(
            sample, treatment_col, outcome_col
        )
        logging.info(f"Patient numbers in sample:\n{sample_table}")

        ps_stats = compute_propensity_score_stats(sample, ps_col, treatment_col)
        logging.info(f"Propensity score stats in sample:\n{ps_stats}")

        compute_effects_for_sample(
            estimators,
            sample,
            results,
            method_args,
            treatment_col,
            outcome_col,
            ps_col,
            **kwargs,
        )

    return process_bootstrap_results(results, n_bootstraps)


def compute_single_effect(
    estimators,
    df,
    treatment_col,
    outcome_col,
    ps_col,
    method_args,
    apply_common_support,
    common_support_threshold,
    **kwargs,
):
    if apply_common_support:
        df = filter_common_support(
            df,
            ps_col=ps_col,
            treatment_col=treatment_col,
            threshold=common_support_threshold,
        )
    initial_table = compute_treatment_outcome_table(df, treatment_col, outcome_col)
    logging.info(f"Patient numbers:\n{initial_table}")

    ps_stats = compute_propensity_score_stats(df, ps_col, treatment_col)
    logging.info(f"Propensity score stats:\n{ps_stats}")

    results = {type(estimator).__name__: [] for estimator in estimators}
    compute_effects_for_sample(
        estimators,
        df,
        results,
        method_args,
        treatment_col,
        outcome_col,
        ps_col,
        **kwargs,
    )

    return process_single_results(results)


def compute_effects_for_sample(
    estimators,
    sample,
    results,
    method_args,
    treatment_col,
    outcome_col,
    ps_col,
    **kwargs,
):
    for estimator in estimators:
        method_name = type(estimator).__name__
        estimator_specific_args = method_args.get(method_name, {})
        effect = estimator.compute_effect(
            sample,
            treatment_col,
            outcome_col,
            ps_col,
            **estimator_specific_args,
            **kwargs,
        )
        results[method_name].append(effect)


def process_bootstrap_results(results, n_bootstraps):
    final_results = {}
    for method_name, effects in results.items():
        effects_array = np.array(effects)
        mean_effect = np.mean(effects_array)
        std_err = np.std(effects_array)
        final_results[method_name] = {
            "effect": mean_effect,
            "std_err": std_err,
            "bootstrap": True,
            "n_bootstraps": n_bootstraps,
        }
    return final_results


def process_single_results(results):
    final_results = {}
    for method_name, effects in results.items():
        final_results[method_name] = {
            "effect": effects[0],
            "std_err": None,
            "bootstrap": False,
            "n_bootstraps": 0,
        }
    return final_results
