from typing import List, Union

import numpy as np
import pandas as pd

from CausalEstimate.core.imports import import_all_estimators
from CausalEstimate.core.registry import ESTIMATOR_REGISTRY
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.utils.checks import check_required_columns, check_columns_for_nans
from CausalEstimate.core.bootstrap import generate_bootstrap_samples

# !TODO: Write test for all functions


class Estimator:
    def __init__(
        self, methods: Union[str, list] = None, effect_type: str = "ATE", **kwargs
    ):
        """
        Initialize the Estimator class with one or more methods.

        Args:
            methods (list or str): A list of estimator method names (e.g., ["AIPW", "TMLE"])
                                   or a single method name (e.g., "AIPW").
            effect_type (str): The type of effect to estimate (e.g., "ATE", "ATT").
            **kwargs: Additional keyword arguments for each estimator.
        """
        if methods is None:
            methods = ["AIPW"]  # Default to AIPW if no method is provided.
        import_all_estimators()
        # Allow single method or list of methods
        self.methods = methods if isinstance(methods, list) else [methods]
        self.effect_type = effect_type
        self.estimators = self._initialize_estimators(effect_type, **kwargs)

    def compute_effect(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ps_col: str,
        bootstrap: bool = False,
        n_bootstraps: int = 100,
        method_args: dict = None,
        apply_common_support: bool = True,
        common_support_threshold: float = 0.05,
        **kwargs,
    ) -> dict:
        """
        Compute treatment effects using the initialized estimators.
        Can also run bootstrap on all estimators if specified.

        Args:
            df (pd.DataFrame): The input DataFrame.
            treatment_col (str): The name of the treatment column.
            outcome_col (str): The name of the outcome column.
            ps_col (str): The name of the propensity score column.
            bootstrap (bool): Whether to run bootstrapping for the estimators.
            n_bootstraps (int): Number of bootstrap iterations.
            method_args (dict): Additional arguments for each estimator.
            apply_common_support (bool): Whether to apply common support filtering.
            common_support_threshold (float): Threshold for common support filtering.
            **kwargs: Additional arguments for the estimators.

        Returns:
            dict: A dictionary where keys are method names and values are computed effects (and optionally standard errors).
        """
        # Validate input data and columns
        self._validate_inputs(df, treatment_col, outcome_col)

        # Initialize results dictionary
        results = {type(estimator).__name__: [] for estimator in self.estimators}

        # Ensure method_args is a dictionary
        method_args = method_args or {}

        if bootstrap:
            # Perform bootstrapping
            bootstrap_samples = generate_bootstrap_samples(df, n_bootstraps)

            for sample in bootstrap_samples:
                # Apply common support filtering if specified
                if apply_common_support:
                    sample = filter_common_support(
                        sample,
                        ps_col=ps_col,
                        treatment_col=treatment_col,
                        threshold=common_support_threshold,
                    )

                # For each bootstrap sample, compute the effect using all estimators
                for estimator in self.estimators:
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

            # After collecting all bootstrap samples, compute the mean and standard error for each estimator
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

        else:
            # If no bootstrapping, apply common support filtering once if specified
            if apply_common_support:
                df = filter_common_support(
                    df,
                    ps_col=ps_col,
                    treatment_col=treatment_col,
                    threshold=common_support_threshold,
                )

            # Compute the effect directly for each estimator
            final_results = {}
            for estimator in self.estimators:
                method_name = type(estimator).__name__
                estimator_specific_args = method_args.get(method_name, {})
                effect = estimator.compute_effect(
                    df,
                    treatment_col,
                    outcome_col,
                    ps_col,
                    **estimator_specific_args,
                    **kwargs,
                )
                final_results[method_name] = {
                    "effect": effect,
                    "std_err": None,
                    "bootstrap": False,
                    "n_bootstraps": 0,
                }

        return final_results

    def _initialize_estimators(self, effect_type: str, **kwargs) -> List[object]:
        """
        Initialize the specified estimators based on the methods provided.
        """
        estimators = []

        for method in self.methods:
            if method not in ESTIMATOR_REGISTRY:
                raise ValueError(f"Method '{method}' is not supported.")
            estimator_class = ESTIMATOR_REGISTRY.get(method)
            estimator = estimator_class(effect_type=effect_type, **kwargs)
            estimators.append(estimator)
        return estimators

    @staticmethod
    def _validate_inputs(df: pd.DataFrame, treatment_col: str, outcome_col: str):
        #!TODO: Move this to base class and individual estimator classes, figure out what else to check and how to better combine it with the checks in the estimators themselves
        """
        Validate the input DataFrame and columns for all estimators.
        """
        check_required_columns(df, [treatment_col, outcome_col])
        check_columns_for_nans(df, [treatment_col, outcome_col])
