# CausalEstimate/estimators/multi_estimator.py

import numpy as np
from typing import List, Dict
import pandas as pd
from CausalEstimate.core.bootstrap import generate_bootstrap_samples
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.estimators.base import BaseEstimator
from CausalEstimate.core.logging import log_table_stats


class MultiEstimator:
    def __init__(self, estimators: List[BaseEstimator], verbose: bool = False):
        """
        `estimators` is a list of estimator instances (AIPW, TMLE, IPW, etc.).
        Each is already configured with its own column names and effect_type.
        """
        self.estimators = estimators
        self.verbose = verbose

    def compute_effects(
        self,
        df: pd.DataFrame,
        bootstrap: bool = False,
        n_bootstraps: int = 1,
        apply_common_support: bool = False,
        common_support_threshold: float = 0.05,
    ) -> Dict[str, Dict]:
        """
        Loops over self.estimators, applies optional common support,
        optional bootstrap, and returns { "AIPW": {...}, "TMLE": {...}, ... }.
        """
        if apply_common_support:
            # Verify all estimators use the same ps_col and treatment_col
            first_estimator = self.estimators[0]
            ps_col = first_estimator.ps_col
            treatment_col = first_estimator.treatment_col
            outcome_col = first_estimator.outcome_col

            for estimator in self.estimators[1:]:
                if (
                    estimator.ps_col != ps_col
                    or estimator.treatment_col != treatment_col
                ):
                    raise ValueError(
                        "All estimators must use the same ps_col and treatment_col "
                        f"but found ps_col={estimator.ps_col} vs {ps_col} and "
                        f"treatment_col={estimator.treatment_col} vs {treatment_col}"
                    )

            df = filter_common_support(
                df,
                ps_col=ps_col,
                treatment_col=treatment_col,
                threshold=common_support_threshold,
            ).reset_index(drop=True)

        if self.verbose:
            log_table_stats(df, treatment_col, outcome_col, ps_col)

        results = {}
        for estimator in self.estimators:
            est_name = estimator.__class__.__name__

            if bootstrap and n_bootstraps > 1:
                effect, std_err = self._compute_bootstrap(estimator, df, n_bootstraps)
                results[est_name] = {
                    "effect": effect,
                    "std_err": std_err,
                    "bootstrap": True,
                    "n_bootstraps": n_bootstraps,
                }
            else:
                effect_val = estimator.compute_effect(df)
                results[est_name] = {
                    "effect": effect_val,
                    "std_err": None,
                    "bootstrap": False,
                    "n_bootstraps": 0,
                }
        return results

    def _compute_bootstrap(
        self, estimator: BaseEstimator, df: pd.DataFrame, n_bootstraps: int
    ):
        effects = []
        samples = generate_bootstrap_samples(df, n_bootstraps)
        for sample in samples:
            val = estimator.compute_effect(sample)
            effects.append(val)
        return float(np.mean(effects)), float(np.std(effects))
