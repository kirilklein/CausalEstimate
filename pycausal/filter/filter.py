import pandas as pd


def filter_by_column(df: pd.DataFrame, column: str, value: int) -> pd.DataFrame:
    """
    Filters the DataFrame by a specifeid column and value.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    return df[df[column] == value].reset_index(drop=True)
