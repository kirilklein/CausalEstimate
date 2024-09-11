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
