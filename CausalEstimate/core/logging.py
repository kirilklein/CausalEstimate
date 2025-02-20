import logging

import pandas as pd

from CausalEstimate.stats.stats import (
    compute_propensity_score_stats,
    compute_treatment_outcome_table,
)
from CausalEstimate.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def log_table_stats(
    df: pd.DataFrame, treatment_col: str, outcome_col: str, ps_col: str
):
    initial_table = compute_treatment_outcome_table(df, treatment_col, outcome_col)
    ps_stats = compute_propensity_score_stats(df, ps_col, treatment_col)
    logging.info(f"Initial patient numbers:\n{initial_table}")
    logging.info(f"Initial propensity score stats:\n{ps_stats}")
