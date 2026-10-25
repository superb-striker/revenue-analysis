# Revenue analysis: pre-production churn pipeline

The model predicts **retention 30 days after the customer's first delivered order**. The target is `churned` (`0` means a second delivered order is purchased within the following 180 days, `1` means none); the quality metric is average precision (PR-AUC) for the retained class. This is a staging workflow, with an explicit quality gate before an artifact can be built into the inference image.

## Data and timing

Use `sql/07_churn_features.sql` to export one row per customer to CSV. The SQL requires a full 180-day observation window after scoring and excludes customers who reordered before scoring. Seller history uses orders delivered before the first purchase; reviews are cut off at scoring. The service must score at **30 days after first delivery**. The training CSV needs `scoring_date`, `churned`, and all columns listed in `revenue_analysis.ml.FEATURES`. Inference requires exactly those feature columns and never accepts the target. No customer identifier or later purchase information enters the model.

Pandera rejects missing required fields, invalid ranges, fractional counts, and inconsistent item/seller counts. Optional missing fields are imputed inside the fitted pipeline. Category buckets and imputers are learned only on the training partition. Training uses a temporal holdout ordered by `scoring_date`, evaluates it once, then refits the approved model on all validated rows. The gate requires retained PR-AUC of at least `max(configured minimum, holdout retained prevalence + 0.05)`. Set the minimum for the deployment context; `0.60` is the default example, not a claim about real Olist performance.

## Train and test

```sh
uv sync --locked
uv run pytest -q
uv run revenue-train --data tests/fixtures/churn.csv --output model --min-pr-auc 0.60
```

Use a CSV exported from the SQL query for a real staging run. A failed quality gate exits with an error before writing the model. The fixture is deliberately synthetic and only exercises the pipeline; its score does not estimate real performance. `model/report.json` contains a deterministic model version, dataset and pipeline code SHA-256 hashes, temporal split details, metrics, training configuration, library versions, feature schema, missing rates, drift reference, and model checksum. Preserve `model.joblib` and `report.json` together; load joblib files only from trusted training runs. The `model/` directory is ignored by Git.

## Staging inference

```sh
docker build -f Dockerfile.staging -t revenue-staging .
docker run --rm -p 8000:8000 revenue-staging
```

The image embeds the `model/` directory generated immediately before building. `/health` reports the model version. `POST /predict` accepts `{"records": [{...one object with every feature column...}]}` and returns retained probabilities, the model version, and per-feature drift diagnostics. Use `revenue_analysis.ml.FEATURES` as the request field list. Inference batches are capped at 1,000 rows. Drift uses PSI for numeric columns and total variation distance for categorical columns; default limits are 0.2, with a 0.1 limit on missing-rate change. Batches under 30 rows are marked as insufficient for a drift decision. Compare representative batches to the reference saved from training, and investigate flagged shifts before promotion.

CI runs all tests, trains the small fixture with the same PR-AUC gate, builds the staging image, and uploads the model report. The existing `python/export_for_tableau.py` remains an analysis export; model promotion should use `revenue-train` and the gated artifact.
