"""Evaluate an XGBoost baseline across the curated OpenML classification suite.

XGBoost is the natural "strong tree baseline" to put next to TabPFN and its distilled students:
gradient-boosted trees are what practitioners actually reach for on small tabular data, so beating
(or matching) XGBoost is the bar a distilled model has to clear.

This reuses ``OpenMLBenchmark`` so the protocol is identical to the other result files in ``results/``
(``StratifiedShuffleSplit`` with ``random_state=0``, accuracy + macro one-vs-rest ROC AUC, repeated
splits). It writes two CSVs in the same shape as the existing summaries:

* ``results/xgboost_dataset.csv`` -- one row per dataset (``dataset, n_rows, n_classes,
  xgboost_acc, xgboost_auc``), the per-dataset mean over splits.
* ``results/xgboost_sum.csv``     -- one row (``model, acc_mean, acc_std, auc_mean, auc_std``),
  averaged across datasets, matching ``*_sum.csv``.

Usage
-----
    uv run python evaluate_xgboost.py                       # full classification suite, 5 splits
    uv run python evaluate_xgboost.py --n-splits 10
    uv run python evaluate_xgboost.py --datasets sonar wine diabetes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from benchmark_datasets import RANDOM_STATE, OpenMLBenchmark

RESULTS_DIR = Path(__file__).parent / "results"


def make_xgb(_task: str) -> XGBClassifier:
    """Fresh XGBoost classifier with solid small-data defaults.

    A fresh estimator is built per split by ``OpenMLBenchmark.evaluate``. ``num_class`` is inferred
    by XGBoost from the (contiguous, label-encoded) targets, so the same factory covers binary and
    multiclass. ``predict_proba`` gives ROC AUC the soft scores it needs.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5, help="repeated shuffle splits per dataset")
    parser.add_argument("--train-size", type=float, default=0.6, help="train fraction per split")
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="subset of dataset names (default: full suite)"
    )
    parser.add_argument(
        "--no-scale", action="store_true", help="disable StandardScaler (harmless for trees)"
    )
    args = parser.parse_args()

    bench = OpenMLBenchmark(task="classification")
    per_dataset = bench.evaluate(
        make_xgb,
        datasets=args.datasets,
        task="classification",
        n_splits=args.n_splits,
        train_size=args.train_size,
        scale=not args.no_scale,
        verbose=True,
    )

    # n_classes per dataset (load is cached on disk, so this is cheap).
    per_dataset["n_classes"] = [
        bench.load(name).n_classes for name in per_dataset["name"]
    ]

    dataset_out = per_dataset.rename(
        columns={"name": "dataset", "acc_mean": "xgboost_acc", "auc_mean": "xgboost_auc"}
    )[["dataset", "n_rows", "n_classes", "xgboost_acc", "xgboost_auc"]]

    summary = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "acc_mean": float(per_dataset["acc_mean"].mean()),
                "acc_std": float(per_dataset["acc_mean"].std()),
                "auc_mean": float(np.nanmean(per_dataset["auc_mean"])),
                "auc_std": float(np.nanstd(per_dataset["auc_mean"])),
            }
        ]
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    dataset_out.to_csv(RESULTS_DIR / "xgboost_dataset.csv")
    summary.to_csv(RESULTS_DIR / "xgboost_sum.csv")

    print("\n=== XGBoost summary (mean across datasets) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'xgboost_dataset.csv'}")
    print(f"Wrote {RESULTS_DIR / 'xgboost_sum.csv'}")


if __name__ == "__main__":
    main()
