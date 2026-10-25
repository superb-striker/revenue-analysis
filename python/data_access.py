from pathlib import Path
import pandas as pd
from sqlalchemy import text
from config import get_engine

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def run_sql_file(filename: str) -> pd.DataFrame:
    engine = get_engine()
    query = (SQL_DIR / filename).read_text()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def load_window_functions() -> pd.DataFrame:
    return run_sql_file("03_window_functions.sql")


def load_cohort_retention() -> pd.DataFrame:
    return run_sql_file("04_cohort_retention.sql")


def load_rfm_segmentation() -> pd.DataFrame:
    return run_sql_file("05_rfm_segmentation.sql")


def load_revenue_leakage() -> pd.DataFrame:
    return run_sql_file("06_revenue_leakage.sql")


def load_churn_features() -> pd.DataFrame:
    return run_sql_file("07_churn_features.sql")
