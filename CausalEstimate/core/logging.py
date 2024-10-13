import logging

import pandas as pd

from CausalEstimate.stats.stats import (
    compute_propensity_score_stats,
    compute_treatment_outcome_table,
)
from CausalEstimate.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def log_sample_stats(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    ps_col: str,
) -> None:
    sample_table = compute_treatment_outcome_table(
        df=df, treatment_col=treatment_col, outcome_col=outcome_col
    )
    ps_stats = compute_propensity_score_stats(
        df=df, ps_col=ps_col, treatment_col=treatment_col
    )
    logging.info(f"Patient numbers in sample:\n{sample_table}")
    logging.info(f"Propensity score stats in sample:\n{ps_stats}")


def log_initial_stats(
    df: pd.DataFrame, treatment_col: str, outcome_col: str, ps_col: str
):
    initial_table = compute_treatment_outcome_table(df, treatment_col, outcome_col)
    ps_stats = compute_propensity_score_stats(df, ps_col, treatment_col)
    logging.info(f"Initial patient numbers:\n{initial_table}")
    logging.info(f"Initial propensity score stats:\n{ps_stats}")
