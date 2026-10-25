import numpy as np
from sklearn.model_selection import train_test_split

from revenue_analysis.ml import FEATURES, FeatureBuilder, pipeline, validate_frame


def test_train_serve_feature_consistency(fixture_frame):
    frame = validate_frame(fixture_frame, training=True)
    X = frame.loc[:, FEATURES]
    train, serve = train_test_split(X, random_state=42)
    model = pipeline(42).fit(train, frame.loc[train.index, "churned"])
    direct = model.predict_proba(serve)
    api_shape = serve.to_dict(orient="records")
    rebuilt = validate_frame(__import__("pandas").DataFrame(api_shape), training=False)
    assert np.allclose(direct, model.predict_proba(rebuilt.loc[:, FEATURES]))
    assert list(model.named_steps["features"].transform(serve)) == list(FEATURES[:15]) + ["has_review"] + list(FEATURES[15:])


def test_unseen_categories_bucketed_without_refit(fixture_frame):
    builder = FeatureBuilder().fit(fixture_frame.loc[:, FEATURES])
    incoming = fixture_frame.loc[[0], FEATURES].copy()
    incoming.loc[0, "product_category"] = "unseen"
    assert builder.transform(incoming).iloc[0]["product_category"] == "Other"
