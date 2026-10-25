"""Validated, point-in-time first-order churn training and scoring."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
import sklearn
from pandera import Check, Column
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = (
    "price", "freight_ratio", "num_items", "num_sellers", "payment_installments",
    "review_score", "delivery_delay_days", "estimated_lead_days",
    "actual_delivery_days", "purchase_dow", "purchase_month", "purchase_hour",
    "seller_late_rate", "seller_order_count", "customer_seller_distance_km",
)
CATEGORICAL = ("product_category", "payment_type", "customer_state")
FEATURES = NUMERIC + CATEGORICAL
DERIVED = ("has_review",)
TARGET = "churned"
TIME_COLUMN = "scoring_date"
SCHEMA_VERSION = 1


def _numeric_schema() -> dict:
    bounds = {
        "price": (0, None), "freight_ratio": (0, 1), "num_items": (1, None),
        "num_sellers": (1, None), "payment_installments": (0, None),
        "review_score": (1, 5), "delivery_delay_days": (-365, 3650),
        "estimated_lead_days": (0, 3650), "actual_delivery_days": (0, 3650),
        "purchase_dow": (0, 6), "purchase_month": (1, 12),
        "purchase_hour": (0, 23), "seller_late_rate": (0, 1),
        "seller_order_count": (0, None), "customer_seller_distance_km": (0, 20050),
    }
    whole_numbers = {"num_items", "num_sellers", "payment_installments", "purchase_dow", "purchase_month", "purchase_hour", "seller_order_count"}
    columns = {}
    for name, (lo, hi) in bounds.items():
        checks = [Check.in_range(lo, hi) if hi is not None else Check.ge(lo)]
        if name in whole_numbers:
            checks.append(Check(lambda s: (s.dropna() % 1 == 0).all(), element_wise=False))
        columns[name] = Column(float, nullable=name in {"review_score", "delivery_delay_days", "estimated_lead_days", "actual_delivery_days", "seller_late_rate", "seller_order_count", "customer_seller_distance_km", "payment_installments", "freight_ratio"}, coerce=True, checks=checks)
    return columns


def validate_frame(frame: pd.DataFrame, *, training: bool) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Input must contain at least one row")
    if not training and TARGET in frame:
        raise ValueError("Inference input must not contain target")
    columns = _numeric_schema()
    columns.update({name: Column(str, nullable=True, coerce=False) for name in CATEGORICAL})
    if training:
        columns[TARGET] = Column(int, Check.isin([0, 1]), coerce=True)
        columns[TIME_COLUMN] = Column(pa.DateTime, nullable=False, coerce=True)
    schema = pa.DataFrameSchema(columns, strict=False,
                                checks=[Check(lambda df: (df["num_sellers"] <= df["num_items"]).all())])
    result = schema.validate(frame, lazy=True)
    if training and result[TARGET].nunique() != 2:
        raise ValueError("Training target must contain both classes")
    # Missingness is explicit: optional fields are imputed, mandatory fields fail.
    return result


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Learns rare category buckets only from the training partition."""

    def fit(self, X, y=None):
        self.top_categories_ = self._top(X["product_category"], 10)
        self.top_states_ = self._top(X["customer_state"], 8)
        return self

    @staticmethod
    def _top(series, n):
        return tuple(series.fillna("Unknown").value_counts().nlargest(n).index)

    def transform(self, X):
        frame = X.loc[:, FEATURES].copy()
        for name, top in (("product_category", self.top_categories_), ("customer_state", self.top_states_)):
            series = frame[name].fillna("Unknown")
            frame[name] = series.where(series.isin(top), "Other")
        frame["payment_type"] = frame["payment_type"].fillna("Unknown")
        frame["has_review"] = frame["review_score"].notna().astype(float)
        return frame.loc[:, NUMERIC + DERIVED + CATEGORICAL]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    test_size: float = 0.25
    min_pr_auc: float = 0.60
    min_ap_lift: float = 0.05
    max_missing_rate: float = 0.50


def pipeline(seed: int) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([
        ("features", FeatureBuilder()),
        ("preprocess", ColumnTransformer([("num", numeric, NUMERIC + DERIVED), ("cat", categorical, CATEGORICAL)])),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])


def dataset_hash(frame: pd.DataFrame) -> str:
    cols = FEATURES + (TARGET, TIME_COLUMN)
    canonical = frame.loc[:, cols].sort_values(list(cols), na_position="first").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).values.tobytes()).hexdigest()


def missing_rates(frame: pd.DataFrame) -> dict[str, float]:
    return {col: float(frame[col].isna().mean()) for col in FEATURES}


def check_quality(pr_auc: float, threshold: float, prevalence: float = 0.0, min_lift: float = 0.0) -> None:
    required = max(threshold, prevalence + min_lift)
    if not np.isfinite(pr_auc) or pr_auc < required:
        raise ValueError(f"PR-AUC quality gate failed: {pr_auc:.4f} < {required:.4f}")


def reference_profile(frame: pd.DataFrame) -> dict:
    profile = {}
    for name in NUMERIC:
        values = frame[name].dropna().astype(float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))) if len(values) else np.array([])
        if len(edges) < 2:
            edges = np.array([float(values.iloc[0]) - 0.5, float(values.iloc[0]) + 0.5]) if len(values) else np.array([0., 1.])
        counts = np.histogram(values, bins=np.r_[-np.inf, edges[1:-1], np.inf])[0]
        profile[name] = {"kind": "numeric", "edges": edges[1:-1].tolist(), "shares": (counts / max(len(values), 1)).tolist(), "missing": float(frame[name].isna().mean())}
    for name in CATEGORICAL:
        counts = frame[name].fillna("<MISSING>").value_counts(normalize=True)
        profile[name] = {"kind": "categorical", "shares": counts.to_dict(), "missing": float(frame[name].isna().mean())}
    return profile


def detect_drift(frame: pd.DataFrame, reference: dict, *, psi_threshold: float = 0.2, tvd_threshold: float = 0.2, missing_delta: float = 0.1) -> dict:
    frame = validate_frame(frame, training=False)
    results = {}
    sufficient = len(frame) >= 30
    for name, baseline in reference.items():
        if baseline["kind"] == "numeric":
            values = frame[name].dropna().to_numpy(dtype=float)
            shares = np.histogram(values, bins=np.r_[-np.inf, baseline["edges"], np.inf])[0] / max(len(values), 1)
            expected = np.asarray(baseline["shares"])
            score = float(np.sum((shares - expected) * np.log((shares + 1e-6) / (expected + 1e-6))))
            limit = psi_threshold
            metric = "psi"
        else:
            actual = frame[name].fillna("<MISSING>").value_counts(normalize=True).to_dict()
            cats = set(actual) | set(baseline["shares"])
            score = sum(abs(actual.get(cat, 0) - baseline["shares"].get(cat, 0)) for cat in cats) / 2
            limit = tvd_threshold
            metric = "tvd"
        missing_change = abs(float(frame[name].isna().mean()) - baseline["missing"])
        results[name] = {metric: score, "missing_delta": missing_change,
                         "sufficient_sample": sufficient,
                         "drifted": bool(sufficient and (score > limit or missing_change > missing_delta))}
    return results


def train(frame: pd.DataFrame, output: Path, config: TrainConfig = TrainConfig()) -> dict:
    frame = validate_frame(frame, training=True)
    if not 0 < config.test_size < 1 or not 0 <= config.min_pr_auc <= 1 or not 0 <= config.min_ap_lift <= 1:
        raise ValueError("Invalid training configuration")
    excessive = {name: rate for name, rate in missing_rates(frame).items() if rate > config.max_missing_rate}
    if excessive:
        raise ValueError(f"Excessive missingness: {excessive}")
    frame = frame.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    cutoff = frame[TIME_COLUMN].iloc[int(len(frame) * (1 - config.test_size))]
    train_rows = frame[frame[TIME_COLUMN] < cutoff]
    test_rows = frame[frame[TIME_COLUMN] >= cutoff]
    if train_rows[TARGET].nunique() != 2 or test_rows[TARGET].nunique() != 2:
        raise ValueError("Temporal holdout must contain both classes in train and test")
    X_train, X_test = train_rows.loc[:, FEATURES], test_rows.loc[:, FEATURES]
    y_train, y_test = train_rows[TARGET].astype(int), test_rows[TARGET].astype(int)
    model = pipeline(config.seed)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 0]  # retained is the rare positive class
    retained = 1 - y_test.to_numpy()
    metrics = {"pr_auc_retained": float(average_precision_score(retained, probabilities)),
               "roc_auc_retained": float(roc_auc_score(retained, probabilities)),
               "retained_prevalence": float(retained.mean()), "holdout_rows": len(y_test), "train_rows": len(y_train)}
    metrics["required_pr_auc"] = max(config.min_pr_auc, metrics["retained_prevalence"] + config.min_ap_lift)
    check_quality(metrics["pr_auc_retained"], config.min_pr_auc, metrics["retained_prevalence"], config.min_ap_lift)
    # The held-out candidate is evaluated before a fresh fit on all data.
    model = pipeline(config.seed).fit(frame.loc[:, FEATURES], frame[TARGET].astype(int))
    profile = reference_profile(frame.loc[:, FEATURES])
    digest = dataset_hash(frame)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    version = hashlib.sha256((digest + code_hash + json.dumps(asdict(config), sort_keys=True)).encode()).hexdigest()[:16]
    report = {"schema_version": SCHEMA_VERSION, "model_version": version, "dataset_sha256": digest,
              "pipeline_sha256": code_hash, "library_versions": versions,
              "metrics": metrics, "config": asdict(config), "feature_columns": list(FEATURES),
              "split": {"method": TIME_COLUMN, "holdout_starts": cutoff.date().isoformat()},
              "training_missing_rates": missing_rates(frame.loc[:, FEATURES]), "reference_profile": profile}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    report["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_model(directory: Path):
    report = json.loads((directory / "report.json").read_text())
    if report["schema_version"] != SCHEMA_VERSION or report["feature_columns"] != list(FEATURES):
        raise ValueError("Incompatible model report")
    if report["pipeline_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Model was built with different pipeline code")
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    if report["library_versions"] != versions:
        raise ValueError("Model was built with incompatible library versions")
    if hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("Model artifact checksum mismatch")
    model = joblib.load(directory / "model.joblib")  # Only load trusted training artifacts.
    return model, report
