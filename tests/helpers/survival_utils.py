import pandas as pd

def transform_data_for_model_estimation(data: pd.DataFrame) -> pd.DataFrame:
    """Transform the data for either the failure model or the censoring model estimation.

    Args:
        data (pd.DataFrame): The input data containing T_observed, Y, A, X.
        model_type (str): Either 'failure' or 'censoring' to specify which model to transform the data for.

    Returns:
        pd.DataFrame: The transformed data suitable for classification.
    """
    expanded_rows = []

    # Loop over each patient in the original data
    for index, row in data.iterrows():
        T = int(row["T_observed"])
        expanded_rows.extend(
            [
                {
                    "pid": index,
                    "t": t,
                    "A": row["A"],
                    "X": row["X"],
                    "Y_E": get_indicator(t, T, row["Y"], "failure"),
                    "Y_C": get_indicator(t, T, row["Y"], "censoring"),
                }
                for t in range(0, T + 1)
            ]
        )

    # Create DataFrame from the list of rows
    expanded_data = pd.DataFrame(expanded_rows)
    return expanded_data

def get_indicator(t: int, T_observed: int, Y: bool, model_type: str) -> int:
    """Determine the event or censoring indicator based on model type."""
    if model_type == "failure":
        return 1 if ((t == T_observed) and (Y == 1)) else 0
    elif model_type == "censoring":
        return 1 if ((t == T_observed) and (Y == 0)) else 0
    else:
        raise ValueError("model_type must be either 'failure' or 'censoring'")
    
