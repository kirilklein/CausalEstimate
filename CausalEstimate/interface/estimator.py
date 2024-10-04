from typing import List, Union

import numpy as np
import pandas as pd

from CausalEstimate.core.imports import import_all_estimators
from CausalEstimate.core.registry import ESTIMATOR_REGISTRY

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

    def _validate_inputs(self, df, treatment_col, outcome_col):
        #!TODO: Move this to base class and individual estimator classes
        """
        Validate the input DataFrame and columns for all estimators.
        """
        required_columns = [treatment_col, outcome_col]
        # Check if all required columns exist in the DataFrame
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from the DataFrame.")

        # Additional validation logic if needed (e.g., check for NaN, etc.)
        if df[treatment_col].isnull().any():
            raise ValueError(f"Treatment column '{treatment_col}' contains NaN values.")
        if df[outcome_col].isnull().any():
            raise ValueError(f"Outcome column '{outcome_col}' contains NaN values.")

    def _bootstrap_sample(
        self, df: pd.DataFrame, n_bootstraps: int
    ) -> List[pd.DataFrame]:
        """
        Generate bootstrap samples.
        """
        n = len(df)
        return [df.sample(n=n, replace=True) for _ in range(n_bootstraps)]

    def compute_effect(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ps_col: str,
        bootstrap: bool = False,
        n_bootstraps: int = 100,
        method_args: dict = None,
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
            sample_size (int): Size of each bootstrap sample.
            method_args (dict): Additional arguments for each estimator.
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
            bootstrap_samples = self._bootstrap_sample(df, n_bootstraps)

            for sample in bootstrap_samples:
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
            # If no bootstrapping, compute the effect directly for each estimator
            final_results = {}
            # If no bootstrapping, compute the effect directly for each estimator
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
