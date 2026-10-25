from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def fixture_frame():
    return pd.read_csv(Path(__file__).parent / "fixtures" / "churn.csv")
