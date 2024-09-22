import pandas as pd
import numpy as np
from CausalEstimate.estimators.tmle import TMLE

# !TODO: Write test for all functions


class Estimator:
    def __init__(self, methods=None, effect_type="ATE", **kwargs):
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

        # Allow single method or list of methods
        self.methods = methods if isinstance(methods, list) else [methods]
        self.effect_type = effect_type
        self.estimators = self._initialize_estimators(effect_type, **kwargs)

    def _initialize_estimators(self, effect_type, **kwargs):
        """
        Initialize the specified estimators based on the methods provided.
        """
        estimators = {
            "TMLE": TMLE,
            # Add other estimators as needed
        }
        initialized_estimators = []

        for method in self.methods:
            if method in estimators:
                initialized_estimators.append(
                    estimators[method](effect_type=effect_type, **kwargs)
                )
            else:
                raise ValueError(f"Method '{method}' is not supported.")

        return initialized_estimators

    def _validate_inputs(self, df, treatment_col, outcome_col):
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

    def _bootstrap_sample(self, df: pd.DataFrame, n_bootstraps: int):
        """
        Generate bootstrap samples.
        """
        n = len(df)
        return [df.sample(n=n, replace=True) for _ in range(n_bootstraps)]

    def compute_effect(
        self,
        df,
        treatment_col,
        outcome_col,
        bootstrap=False,
        n_bootstraps=100,
        **kwargs,
    ):
        """
        Compute treatment effects using the initialized estimators.
        Can also run bootstrap on all estimators if specified.

        Args:
            df (pd.DataFrame): The input DataFrame.
            treatment_col (str): The name of the treatment column.
            outcome_col (str): The name of the outcome column.
            bootstrap (bool): Whether to run bootstrapping for the estimators.
            n_bootstraps (int): Number of bootstrap iterations.
            sample_size (int): Size of each bootstrap sample.
            **kwargs: Additional arguments for the estimators.

        Returns:
            dict: A dictionary where keys are method names and values are computed effects (and optionally standard errors).
        """
        # Validate input data and columns
        self._validate_inputs(df, treatment_col, outcome_col)

        results = {method.__class__.__name__: [] for method in self.estimators}

        if bootstrap:
            # Perform bootstrapping
            bootstrap_samples = self._bootstrap_sample(df, n_bootstraps)

            for sample in bootstrap_samples:
                # For each bootstrap sample, compute the effect using all estimators
                for estimator in self.estimators:
                    method_name = type(estimator).__name__
                    effect = estimator.compute_effect(
                        sample, treatment_col, outcome_col, **kwargs
                    )
                    results[method_name].append(effect)

            # After collecting all bootstrap samples, compute the mean and standard error for each estimator
            final_results = {}
            for method_name, effects in results.items():
                effects_array = np.array(effects)
                mean_effect = np.mean(effects_array)
                std_err = np.std(effects_array)
                final_results[method_name] = (mean_effect, std_err)

        else:
            # If no bootstrapping, compute the effect directly for each estimator
            for estimator in self.estimators:
                method_name = type(estimator).__name__
                effect = estimator.compute_effect(
                    df, treatment_col, outcome_col, **kwargs
                )
                results[method_name] = effect

            final_results = results

        return final_results
