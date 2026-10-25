import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from revenue_analysis.ml import validate_frame


def test_valid_fixture_and_optional_missing(fixture_frame):
    validated = validate_frame(fixture_frame, training=True)
    assert len(validated) == len(fixture_frame)
    assert validated.review_score.isna().any()


@pytest.mark.parametrize("column,value", [("price", -1), ("review_score", 6), ("purchase_month", 13), ("seller_late_rate", 1.1), ("num_items", None), ("num_items", 1.5)])
def test_range_and_required_missing_rejected(fixture_frame, column, value):
    if value == 1.5:
        fixture_frame[column] = fixture_frame[column].astype(float)
    fixture_frame.loc[0, column] = value
    with pytest.raises(SchemaErrors):
        validate_frame(fixture_frame, training=True)


def test_missing_column_and_single_class_rejected(fixture_frame):
    with pytest.raises(SchemaErrors):
        validate_frame(fixture_frame.drop(columns="price"), training=True)
    fixture_frame.churned = 1
    with pytest.raises(ValueError):
        validate_frame(fixture_frame, training=True)


def test_training_requires_event_time(fixture_frame):
    with pytest.raises(SchemaErrors):
        validate_frame(fixture_frame.drop(columns="scoring_date"), training=True)
