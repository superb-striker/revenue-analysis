"""
Runs the full Python-layer pipeline:
  1. Pull the cohort retention, RFM, and revenue leakage tables straight
     from Postgres (SQL already did the heavy lifting there).
  2. Pull the churn feature set and run significance tests on the
     friction-point comparisons and the RFM segment breakdown.
  3. Compare logistic regression, random forest, and gradient boosting
     via cross-validation (primary metric: PR-AUC on the RETAINED class,
     since that's the rare class that plain accuracy/ROC-AUC can hide
     problems for), auto-pick the best, refit it on all data, and score
     every customer.
  4. Write everything Tableau needs as clean CSVs into tableau_exports/.

Run from the python/ folder: `python main.py`
Note: step 3 now fits 15 models total (3 candidates x 5 CV folds), so
this will take noticeably longer than the single-logistic-regression
version -- expect a few minutes on the full ~66k-row churn feature set,
not seconds.
"""
from pathlib import Path
import pandas as pd

from data_access import (
    load_cohort_retention,
    load_rfm_segmentation,
    load_revenue_leakage,
    load_churn_features,
)
from churn_model import compare_models, summarize_comparison, train_final_model, score_all_customers
from stats_tests import two_proportion_ztest, chi_square_segment_test

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tableau_exports"


def export_all():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Pulling cohort retention...")
    cohort_df = load_cohort_retention()
    cohort_df.to_csv(OUTPUT_DIR / "cohort_retention.csv", index=False)
    print(f"  -> {len(cohort_df)} rows")

    print("Pulling RFM segmentation...")
    rfm_df = load_rfm_segmentation()
    rfm_df.to_csv(OUTPUT_DIR / "rfm_segmentation.csv", index=False)
    print(f"  -> {len(rfm_df)} customers")

    print("Pulling revenue leakage summary...")
    leakage_df = load_revenue_leakage()
    leakage_df.to_csv(OUTPUT_DIR / "revenue_leakage.csv", index=False)
    print(f"  -> {len(leakage_df)} friction types")

    print("Pulling churn feature set (this is the biggest pull)...")
    churn_df = load_churn_features()
    print(f"  -> {len(churn_df)} customers with a fair reorder window")

    # ---- Statistical tests ----
    print("\nRunning significance tests...")
    stats_rows = []

    late_result = two_proportion_ztest(churn_df, "is_late_delivery", "churned")
    stats_rows.append({
        "test": "two_proportion_ztest",
        "comparison": "late_delivery_vs_on_time",
        "group_a_rate": late_result["group_true_rate"],
        "group_b_rate": late_result["group_false_rate"],
        "group_a_n": late_result["group_true_n"],
        "group_b_n": late_result["group_false_n"],
        "p_value": late_result["p_value"],
        "significant_at_0.05": late_result["significant_at_0.05"],
    })

    churn_df["is_bad_review"] = (churn_df["review_score"] <= 2).astype(int)
    review_result = two_proportion_ztest(churn_df, "is_bad_review", "churned")
    stats_rows.append({
        "test": "two_proportion_ztest",
        "comparison": "bad_review_vs_not",
        "group_a_rate": review_result["group_true_rate"],
        "group_b_rate": review_result["group_false_rate"],
        "group_a_n": review_result["group_true_n"],
        "group_b_n": review_result["group_false_n"],
        "p_value": review_result["p_value"],
        "significant_at_0.05": review_result["significant_at_0.05"],
    })

    merged = churn_df.merge(
        rfm_df[["customer_unique_id", "rfm_segment"]],
        on="customer_unique_id",
        how="inner",
    )
    chi2_result = chi_square_segment_test(merged, "rfm_segment", "churned")
    stats_rows.append({
        "test": "chi_square",
        "comparison": "rfm_segment_vs_churned",
        "group_a_rate": None,
        "group_b_rate": None,
        "group_a_n": len(merged),
        "group_b_n": None,
        "p_value": chi2_result["p_value"],
        "significant_at_0.05": chi2_result["significant_at_0.05"],
    })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_DIR / "significance_tests_summary.csv", index=False)
    print(stats_df.to_string(index=False))

    segment_churn = (
        merged.groupby("rfm_segment")["churned"]
        .agg(customers="count", churn_rate="mean")
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    segment_churn.to_csv(OUTPUT_DIR / "rfm_segment_churn_rates.csv", index=False)

    # ---- Churn model comparison ----
    print("\nComparing churn models (logistic regression / random forest / "
          "gradient boosting / XGBoost / LightGBM) via 5-fold cross-validation...")
    print("This is the slow step -- 25 model fits total. Grab a coffee, maybe two.")
    cv_results = compare_models(churn_df)
    comparison_summary = summarize_comparison(cv_results)
    comparison_summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (ranked by PR-AUC on the RETAINED class):")
    print(comparison_summary.to_string(index=False))

    best_model_name = comparison_summary.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")
    best_report = cv_results[best_model_name]["classification_report"]
    print("Out-of-fold classification report at F1-optimal threshold:")
    print(pd.DataFrame(best_report).T.to_string())

    # Refit the winning model on ALL data for final scoring/export
    print(f"\nRefitting {best_model_name} on the full dataset...")
    final_pipeline, importance_df = train_final_model(churn_df, best_model_name)
    importance_df.to_csv(OUTPUT_DIR / "churn_model_feature_importance.csv", index=False)

    print("Scoring all customers...")
    churn_df["churn_risk_score"] = score_all_customers(final_pipeline, churn_df)
    churn_df["best_model_used"] = best_model_name
    churn_df.to_csv(OUTPUT_DIR / "churn_customer_features.csv", index=False)

    print(f"\nAll exports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    export_all()
