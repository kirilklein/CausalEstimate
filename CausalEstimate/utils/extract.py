import pandas as pd
from typing import Tuple


def extract_variables(df: pd.DataFrame, *col_names) -> Tuple[pd.Series, ...]:
    return tuple(df[col_name] for col_name in col_names)
