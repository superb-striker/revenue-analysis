import numpy as np
import pytest

from revenue_analysis.ml import FEATURES, load_model, train


def test_reload_equivalence(fixture_frame, tmp_path):
    train(fixture_frame, tmp_path)
    model, report = load_model(tmp_path)
    before = model.predict_proba(fixture_frame.loc[:, FEATURES])
    reloaded, reloaded_report = load_model(tmp_path)
    assert report == reloaded_report
    np.testing.assert_array_equal(before, reloaded.predict_proba(fixture_frame.loc[:, FEATURES]))


def test_corrupted_artifact_rejected_before_deserialization(fixture_frame, tmp_path):
    train(fixture_frame, tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model(tmp_path)
