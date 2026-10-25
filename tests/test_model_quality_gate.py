from dataclasses import replace
import numpy as np
import pytest

from revenue_analysis.ml import TrainConfig, check_quality, train


def test_fixture_passes_gate_and_report(fixture_frame, tmp_path):
    report = train(fixture_frame, tmp_path, TrainConfig(min_pr_auc=.60))
    assert report["metrics"]["pr_auc_retained"] >= .60
    assert len(report["dataset_sha256"]) == 64
    assert report["config"]["seed"] == 42
    assert report["split"]["method"] == "scoring_date"
    assert report["split"]["holdout_starts"] == "2020-07-29"


def test_bad_model_rejected(fixture_frame, tmp_path):
    rng = np.random.default_rng(23)
    fixture_frame["churned"] = rng.permutation(fixture_frame["churned"])
    with pytest.raises(ValueError, match="quality gate failed"):
        train(fixture_frame, tmp_path, TrainConfig(min_pr_auc=.60))
    assert not (tmp_path / "model.joblib").exists()


def test_fixed_seed_reproduces_metrics_and_predictions(fixture_frame, tmp_path):
    first = train(fixture_frame, tmp_path / "one")
    second = train(fixture_frame, tmp_path / "two")
    assert first == second
    assert (tmp_path / "one/report.json").read_bytes() == (tmp_path / "two/report.json").read_bytes()
