"""Train from a CSV and write a versioned artifact directory."""
import argparse
from pathlib import Path

import pandas as pd

from revenue_analysis.ml import TrainConfig, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pr-auc", type=float, default=0.60)
    args = parser.parse_args()
    report = train(pd.read_csv(args.data), args.output, TrainConfig(seed=args.seed, min_pr_auc=args.min_pr_auc))
    print(f"model_version={report['model_version']} pr_auc_retained={report['metrics']['pr_auc_retained']:.4f}")


if __name__ == "__main__":
    main()
