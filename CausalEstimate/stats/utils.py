import pandas as pd


def dataframe_to_nested_dict(df: pd.DataFrame) -> dict:
    """
    Convert dataframe to nested dict.
    Example:
    Input:
        df = pd.DataFrame({'A': [1, 2], 'B': [4, 5]}, index=['X', 'Y'])
    Output:
        {'X': {'A': 1, 'B': 4}, 'Y': {'A': 2, 'B': 5}}
    """
    return df.to_dict(orient="index")
