import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def estimate_survival_probability_at_T(data: pd.DataFrame, model: object) -> np.ndarray:
    """
    Estimate the survival probability at the final time for each patient for a given model (censoring or event).
    Args:
        data (pd.DataFrame): The input data containing T_observed, Y, A, X.
        model (object): The fitted censoring/failure model.

    Returns:
        np.ndarray: The estimated survival probabilities for each patient.
    """
    # Prepare the input features for prediction
    X = data[["t", "A", "X"]]

    # Predict the failure probabilities
    predicted_failure_probas = model.predict_proba(X)[:, 1]

    # Create a new DataFrame for predictions to avoid modifying the original data
    predictions = pd.DataFrame(
        {"pid": data["pid"], "predicted_failure_proba": predicted_failure_probas}
    )

    # Group by 'pid' and compute the survival probability
    # survival_probs = predictions.groupby('pid')['predicted_failure_proba'].apply(lambda x: np.prod(1 - x))
    survival_probs = compute_survival_probabilities(predictions)
    return survival_probs.values

def compute_survival_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    """
    Compute the survival probabilities for each patient.
    """
    return (
        (1 - predictions.groupby("pid")["predicted_failure_proba"].transform("cumprod"))
        .groupby(predictions["pid"])
        .last()
    )

def fit_ps_model(data: pd.DataFrame, model: object = GradientBoostingClassifier()) -> pd.DataFrame:
    """
    Estimate propensity scores.
    """
    X = data["X"].values.reshape(-1, 1)
    treatment_model = model.fit(X, data["A"])
    data["propensity"] = treatment_model.predict_proba(X)[:, 1]
    return data

def fit_failure_model(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a failure model.
    """
    return _fit_temporal_classifier(data, "Y_E")

def fit_censoring_model(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a censoring model.
    """
    return _fit_temporal_classifier(data, "Y_C")

def _fit_temporal_classifier(
    cls_data: pd.DataFrame, target: str, model: object = GradientBoostingClassifier()
):
    """Fit a classifier with the target as the label.
    Target should be one of 'Y_E' or 'Y_C'.
    """
    if target not in ["Y_E", "Y_C"]:
        raise ValueError("target should be either 'Y_E' or 'Y_C'")
    X = cls_data[["t", "A", "X"]]
    Y = cls_data[target]
    return model.fit(X, Y)
