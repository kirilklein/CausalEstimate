import numpy as np


def check_required_columns(df, required_columns):
    """
    Check if all required columns are present in the DataFrame.

    Parameters:
    - df: The input DataFrame to check.
    - required_columns: A list of column names that are required.

    Raises:
    - ValueError: If any required column is missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def check_unique_pid(df, pid_col):
    """
    Check if the input DataFrame is valid.
    PIDS should be unique

    Parameters:
    - df: The input DataFrame to check.
    - pid_col: Column name containing the unique identifier (e.g., patient ID).

    Raises:
    - ValueError: If dataframe contains duplicate PIDs.
    """
    if df[pid_col].nunique() != len(df):
        raise ValueError("Input DataFrame contains duplicate PIDs.")


def check_inputs(
    A=None, Y=None, ps=None, Yhat=None, Y0_hat=None, Y1_hat=None, variable_names=None
):
    """
    Validate inputs for estimators.

    Args:
        A (array-like, optional): Treatment assignment vector.
        Y (array-like, optional): Outcome vector.
        ps (array-like, optional): Propensity score vector.
        Yhat (array-like, optional): Predicted outcome vector.
        Y0_hat (array-like, optional): Predicted outcome under control.
        Y1_hat (array-like, optional): Predicted outcome under treatment.
        variable_names (dict, optional): Mapping from variable to custom name for error messages.
    Raises:
        ValueError: If any of the inputs do not meet the specified conditions.
    """
    # Map variables to their names for error messages
    variable_names = variable_names or {
        "A": "A (Treatment)",
        "Y": "Y (Outcome)",
        "ps": "ps (Propensity Score)",
        "Yhat": "Yhat (Predicted Outcome)",
        "Y0_hat": "Y0_hat (Predicted Outcome under Control)",
        "Y1_hat": "Y1_hat (Predicted Outcome under Treatment)",
    }

    check_binary_array(A, variable_names["A"])
    check_binary_array(Y, variable_names["Y"])
    check_probability_array(ps, variable_names["ps"])

    # Check Yhat (predicted outcome)
    if Yhat is not None:
        check_probability_array(Yhat, variable_names["Yhat"])

    # Check Y0_hat (predicted outcome under control)
    if Y0_hat is not None:
        check_probability_array(Y0_hat, variable_names["Y0_hat"])

    # Check Y1_hat (predicted outcome under treatment)
    if Y1_hat is not None:
        check_probability_array(Y1_hat, variable_names["Y1_hat"])


def check_binary_array(arr, name):
    """Check if all values in the array are binary (0 or 1, including 0.0 and 1.0)"""
    arr = np.asarray(arr)

    # Ensure array has an appropriate numeric or boolean type
    if not (
        np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.bool_)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be of integer, boolean, or floating-point type.")

    # Check that all unique values are either 0 or 1 (including floats)
    unique_values = np.unique(arr)
    if not set(unique_values).issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(f"{name} must contain only 0/1 or True/False values.")


def check_probability_array(arr, name):
    """Check if all values in the array are between 0 and 1 inclusive"""
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")
    if not np.logical_and(arr >= 0, arr <= 1).all():
        raise ValueError(f"{name} must have values between 0 and 1 inclusive.")
