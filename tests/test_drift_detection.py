from revenue_analysis.ml import FEATURES, detect_drift, reference_profile


def test_drift_detection_flags_shift_and_missingness(fixture_frame):
    baseline = fixture_frame.loc[:, FEATURES].copy()
    profile = reference_profile(baseline)
    normal = detect_drift(baseline, profile)
    assert not any(item["drifted"] for item in normal.values())
    shifted = baseline.copy()
    shifted["price"] = shifted["price"] * 10
    shifted["payment_type"] = "new_method"
    shifted["review_score"] = None
    findings = detect_drift(shifted, profile)
    assert findings["price"]["drifted"]
    assert findings["payment_type"]["drifted"]
    assert findings["review_score"]["drifted"]
