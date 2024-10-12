from typing import Dict, List, Union

import pandas as pd

from CausalEstimate.core.effect_computation import compute_effects
from CausalEstimate.core.imports import import_all_estimators
from CausalEstimate.core.registry import ESTIMATOR_REGISTRY
from CausalEstimate.utils.checks import check_columns_for_nans, check_required_columns

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
        n_bootstraps: int = 1,
        apply_common_support: bool = True,
        common_support_threshold: float = 0.05,
        method_args: dict = None,
        **kwargs,
    ) -> Dict[str, Dict]:
        """
        Compute the causal effect using the specified estimators.
        For documentation on the arguments, see the compute_effects function in CausalEstimate.core.effect_computation.
        """
        self._validate_inputs(df, treatment_col, outcome_col)
        return compute_effects(
            estimators=self.estimators,
            df=df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            ps_col=ps_col,
            bootstrap=bootstrap,
            n_bootstraps=n_bootstraps,
            method_args=method_args,
            apply_common_support=apply_common_support,
            common_support_threshold=common_support_threshold,
            **kwargs,
        )

    def _initialize_estimators(self, effect_type: str, **kwargs) -> List[object]:
        """
        Initialize the specified estimators based on the methods provided.
        """
        estimators = []

        for method in self.methods:
            if method not in ESTIMATOR_REGISTRY:
                raise ValueError(f"Method '{method}' is not supported.")
            estimator_class = ESTIMATOR_REGISTRY.get(method.upper())
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
