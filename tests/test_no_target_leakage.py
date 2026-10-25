from pathlib import Path

import pytest

from revenue_analysis.ml import FEATURES, FeatureBuilder, validate_frame


def test_target_and_future_fields_excluded(fixture_frame):
    frame = fixture_frame.copy()
    frame["delivered_order_count"] = frame["churned"] + 1
    frame["second_order_date"] = "2020-01-01"
    X = FeatureBuilder().fit(frame).transform(frame)
    assert "churned" not in X and "delivered_order_count" not in X and "second_order_date" not in X
    assert set(FEATURES).issubset(frame.columns)
    with pytest.raises(ValueError):
        validate_frame(frame, training=False)


def test_seller_history_is_point_in_time():
    sql = (Path(__file__).resolve().parents[1] / "sql/07_churn_features.sql").read_text()
    assert "o.order_delivered_customer_date < fo.first_order_ts" in sql
    assert "r.review_creation_date < fo.scoring_date" in sql
    assert "next_order.next_order_ts > fo.scoring_date" in sql
