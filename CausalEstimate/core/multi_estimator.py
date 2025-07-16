from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from CausalEstimate.core.bootstrap import generate_bootstrap_samples
from CausalEstimate.core.logging import log_table_stats
from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.utils.constants import (
    EFFECT,
    EFFECT_treated,
    EFFECT_untreated,
    INITIAL_EFFECT_treated,
    INITIAL_EFFECT_untreated,
    ADJUSTMENT_treated,
    ADJUSTMENT_untreated,
)


class MultiEstimator:
    def __init__(self, estimators: List[BaseEstimator], verbose: bool = False):
        """
        `estimators` is a list of estimator instances (AIPW, TMLE, IPW, etc.).
        Each is already configured with its own column names and effect_type.
        """
        self.estimators = estimators
        self.verbose = verbose

    def _validate_common_support(
        self, df: pd.DataFrame, common_support_threshold: float
    ) -> Tuple[pd.DataFrame, str, str, str]:
        """
        Ensures all estimators use the same propensity score and treatment columns.
        Filters the DataFrame based on common support.
        Returns the filtered DataFrame along with ps_col, treatment_col, and outcome_col.
        """
        first_estimator = self.estimators[0]
        ps_col = first_estimator.ps_col
        treatment_col = first_estimator.treatment_col
        outcome_col = first_estimator.outcome_col

        for estimator in self.estimators[1:]:
            if estimator.ps_col != ps_col or estimator.treatment_col != treatment_col:
                raise ValueError(
                    "All estimators must use the same ps_col and treatment_col "
                    f"but found ps_col={estimator.ps_col} vs {ps_col} and "
                    f"treatment_col={estimator.treatment_col} vs {treatment_col}"
                )

        filtered_df = filter_common_support(
            df,
            ps_col=ps_col,
            treatment_col=treatment_col,
            threshold=common_support_threshold,
        ).reset_index(drop=True)
        return filtered_df, ps_col, treatment_col, outcome_col

    @staticmethod
    def _safe_mean(values: List) -> float | None:
        """Computes mean of a list, ignoring None values."""
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
        return float(np.mean(valid_values))

    @staticmethod
    def _compute_ci(effects: List[float]) -> Tuple[float, float]:
        """
        Computes the 95% confidence interval using the percentile method.
        Returns the lower and upper bounds.
        """
        lower = float(np.percentile(effects, 2.5))
        upper = float(np.percentile(effects, 97.5))
        return lower, upper

    def compute_effects(
        self,
        df: pd.DataFrame,
        n_bootstraps: int = 1,
        apply_common_support: bool = False,
        common_support_threshold: float = 0.05,
        return_bootstrap_samples: bool = False,
    ) -> Dict[str, Dict]:
        """
        Loops over self.estimators, applies optional common support and bootstrap,
        and returns a dictionary with each estimator's results.

        When bootstrapping is enabled (n_bootstraps > 1), each estimator's output will include:
          - effect: the mean effect across bootstrap samples
          - std_err: the standard deviation of the bootstrap effects
          - CI95_lower and CI95_upper: the 95% confidence interval (using the percentile method)
          - Optionally, raw bootstrap estimates under 'bootstrap_samples' if return_bootstrap_samples is True.
        """
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be at least 1.")

        if apply_common_support:
            df, ps_col, treatment_col, outcome_col = self._validate_common_support(
                df, common_support_threshold
            )
        else:
            first_estimator = self.estimators[0]
            ps_col = first_estimator.ps_col
            treatment_col = first_estimator.treatment_col
            outcome_col = first_estimator.outcome_col

        if self.verbose:
            log_table_stats(df, treatment_col, outcome_col, ps_col)

        results = {}
        for estimator in self.estimators:
            est_name = estimator.__class__.__name__
            if n_bootstraps > 1:
                est_results = self._compute_bootstrap(
                    estimator, df, n_bootstraps, return_bootstrap_samples
                )
                est_results["n_bootstraps"] = n_bootstraps
            else:
                est_results = estimator.compute_effect(df)
                est_results["n_bootstraps"] = 0
            results[est_name] = est_results

        return results

    def _compute_bootstrap(
        self,
        estimator: BaseEstimator,
        df: pd.DataFrame,
        n_bootstraps: int,
        return_bootstrap_samples: bool,
    ) -> Dict[str, Any]:
        """
        Performs bootstrap resampling for a given estimator and returns summary statistics.
        """
        result_keys = [
            EFFECT,
            EFFECT_treated,
            EFFECT_untreated,
            INITIAL_EFFECT_treated,
            INITIAL_EFFECT_untreated,
            ADJUSTMENT_treated,
            ADJUSTMENT_untreated,
        ]
        bootstrap_results = {key: [] for key in result_keys}

        samples = generate_bootstrap_samples(df, n_bootstraps)

        for sample in samples:
            result = estimator.compute_effect(sample)
            for key in result_keys:
                bootstrap_results[key].append(result.get(key))

        effects = bootstrap_results[EFFECT]
        mean_effect = float(np.mean(effects))
        std_err = float(np.std(effects))
        ci95_lower, ci95_upper = self._compute_ci(effects)

        summary: Dict[str, Any] = {
            EFFECT: mean_effect,
            "std_err": std_err,
            "CI95_lower": ci95_lower,
            "CI95_upper": ci95_upper,
        }

        other_keys = [key for key in result_keys if key != EFFECT]
        for key in other_keys:
            summary[key] = self._safe_mean(bootstrap_results[key])

        if return_bootstrap_samples:
            summary["bootstrap_samples"] = {
                EFFECT: bootstrap_results[EFFECT],
                EFFECT_treated: bootstrap_results[EFFECT_treated],
                EFFECT_untreated: bootstrap_results[EFFECT_untreated],
            }
        return summary
