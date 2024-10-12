import pandas as pd

def get_treated_ps(df: pd.DataFrame, treatment_col: str, ps_col: str) -> pd.Series:
    return get_treated(df, treatment_col)[ps_col]

def get_untreated_ps(df: pd.DataFrame, treatment_col: str, ps_col: str) -> pd.Series:
    return get_untreated(df, treatment_col)[ps_col]

def get_treated(df: pd.DataFrame, treatment_col: str) -> pd.DataFrame:
    return df[df[treatment_col] == 1]

def get_untreated(df: pd.DataFrame, treatment_col: str) -> pd.DataFrame:
    return df[df[treatment_col] == 0]