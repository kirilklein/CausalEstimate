import pandas as pd
from scipy.stats import ks_2samp
from CausalEstimate.utils.utils import get_treated_ps, get_untreated_ps


def compute_treatment_outcome_table(
    df: pd.DataFrame, treatment_col: str, outcome_col: str
) -> pd.DataFrame:
    """
    Compute a 2x2 contingency table for binary treatment and outcome.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    treatment_col (str): The name of the column containing the binary treatment indicator
    outcome_col (str): The name of the column containing the binary outcome indicator

    Returns:
    pd.DataFrame: A 2x2 contingency table with rows as treatment (0/1) and columns as outcome (0/1)
    """
    table = pd.crosstab(
        df[treatment_col], df[outcome_col], margins=True, margins_name="Total"
    )
    table.index = ["Untreated", "Treated", "Total"]
    table.columns = ["No Outcome", "Outcome", "Total"]
    return table


def compute_propensity_score_stats(
    df: pd.DataFrame, ps_col: str, treatment_col: str
) -> dict:
    """
    Compare the propensity score distributions between treated and untreated groups.

    This function performs a Kolmogorov-Smirnov test to compare the distributions
    of propensity scores between the treated and untreated groups. It also
    calculates basic statistics for each group.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    ps_col (str): The name of the column containing the propensity scores
    treatment_col (str): The name of the column containing the binary treatment indicator

    Returns:
    dict: A dictionary containing the KS test results and summary statistics
    """
    treated = get_treated_ps(df, treatment_col, ps_col)
    untreated = get_untreated_ps(df, treatment_col, ps_col)

    ks_statistic, p_value = ks_2samp(treated, untreated)

    return {"ks_statistic": ks_statistic, "p_value": p_value}
