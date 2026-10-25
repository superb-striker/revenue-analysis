"""
Churn model comparison: logistic regression, random forest, sklearn
gradient boosting, XGBoost, and LightGBM -- evaluated properly for a
heavily imbalanced target.

Every model uses balanced class weighting. Primary comparison metric is
PR-AUC (average precision) with RETAINED as the positive class, since
that's the rare class and the one accuracy/plain-ROC-AUC can hide
problems for.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from revenue_analysis.ml import FeatureBuilder, FEATURES, validate_frame

TOP_N_CATEGORIES = 10
TOP_N_STATES = 8

NUMERIC_FEATURES = [
    "price", "freight_ratio", "num_items", "num_sellers",
    "payment_installments", "review_score", "has_review",
    "delivery_delay_days", "estimated_lead_days", "actual_delivery_days",
    "purchase_dow", "purchase_month", "purchase_hour",
    "seller_late_rate", "seller_order_count", "customer_seller_distance_km",
]
CATEGORICAL_FEATURES = ["product_category", "payment_type", "customer_state"]


def _bucket_rare(series: pd.Series, top_n: int, other_label: str = "Other") -> pd.Series:
    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), other_label)


def prepare_features(df: pd.DataFrame):
    df = validate_frame(df, training=True)
    X = df.loc[:, FEATURES]
    y = df["churned"].astype(int)
    return X, y, NUMERIC_FEATURES


def build_preprocessor(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def _make_estimator(model_name: str, random_state: int, scale_pos_weight: float = 1.0):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=15,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=random_state,
        )
    if model_name == "xgboost":
        # scale_pos_weight scales class 1 (churned=MAJORITY here) relative to
        # class 0 (retained=minority); passing (#retained / #churned), a
        # value < 1, correctly down-weights the majority class.
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
            eval_metric="logloss", n_jobs=-1,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=random_state, verbosity=-1,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _make_pipeline(model_name: str, numeric_features, random_state: int, scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline(steps=[
        ("features", FeatureBuilder()),
        ("preprocess", build_preprocessor(numeric_features)),
        ("model", _make_estimator(model_name, random_state, scale_pos_weight)),
    ])


def best_threshold_for_f1(y_true_retained: np.ndarray, proba_retained: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true_retained, proba_retained)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.zeros_like(precision), where=(precision + recall) > 0)
    best_idx = np.argmax(f1[:-1]) if len(thresholds) else 0
    return thresholds[best_idx] if len(thresholds) else 0.5


def compare_models(df: pd.DataFrame, model_names=None, n_splits: int = 5, random_state: int = 42):
    if model_names is None:
        model_names = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]

    X, y, numeric_features = prepare_features(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    results = {}
    for model_name in model_names:
        oof_proba_churn = np.zeros(len(y))
        for train_idx, val_idx in skf.split(X, y):
            pipeline = _make_pipeline(model_name, numeric_features, random_state, scale_pos_weight)
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_proba_churn[val_idx] = pipeline.predict_proba(X.iloc[val_idx])[:, 1]

        proba_retained = 1 - oof_proba_churn
        y_retained = 1 - y.values

        roc_auc = roc_auc_score(y, oof_proba_churn)
        pr_auc_retained = average_precision_score(y_retained, proba_retained)

        threshold = best_threshold_for_f1(y_retained, proba_retained)
        y_pred_retained = (proba_retained >= threshold).astype(int)
        y_pred_churn = 1 - y_pred_retained

        report = classification_report(y, y_pred_churn, output_dict=True)
        cm = confusion_matrix(y, y_pred_churn)

        results[model_name] = {
            "roc_auc": roc_auc,
            "pr_auc_retained": pr_auc_retained,
            "threshold_on_p_retained": threshold,
            "classification_report": report,
            "confusion_matrix": cm,
            "oof_proba_churn": oof_proba_churn,
        }
    return results


def summarize_comparison(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "roc_auc": r["roc_auc"],
            "pr_auc_retained": r["pr_auc_retained"],
            "retained_precision": r["classification_report"]["0"]["precision"],
            "retained_recall": r["classification_report"]["0"]["recall"],
            "retained_f1": r["classification_report"]["0"]["f1-score"],
        })
    return pd.DataFrame(rows).sort_values("pr_auc_retained", ascending=False)


def train_final_model(df: pd.DataFrame, model_name: str, random_state: int = 42):
    X, y, numeric_features = prepare_features(df)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    pipeline = _make_pipeline(model_name, numeric_features, random_state, scale_pos_weight)
    pipeline.fit(X, y)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        importance = model.coef_[0]
        importance_df = pd.DataFrame({
            "feature": feature_names, "importance": importance, "odds_ratio": np.exp(importance),
        }).sort_values("importance", ascending=False)
    else:
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names, "importance": importance,
        }).sort_values("importance", ascending=False)

    return pipeline, importance_df


def score_all_customers(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X, _, _ = prepare_features(df)
    return pipeline.predict_proba(X)[:, 1]
