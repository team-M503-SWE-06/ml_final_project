"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

Steps:
  1. load + clean the data (cleaning log printed),
  2. framing + frozen stratified train/val/test split (data/split_indices.npz),
  3. EDA figures,
  4. tree sweeps: criterion, max_depth, min_samples_leaf  (validation only),
  5. SVM sweeps: lambda, RFF bandwidth, objective curve  (validation only),
  6. SVM analysis: weights, epochs/schedule, matched sklearn comparison,
  7. benchmark vs scikit-learn: 5-fold CV mean ± std, timings, one test-set
     evaluation, tree structure comparison,
  8. final results table.

Figures go to report/figures/, tables (CSV) to report/tables/. Every random
step is seeded (data_prep.SEED), so reruns give the same numbers.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from . import data_prep, eda
from . import evaluate as ev
from .experiments import benchmark, svm_analysis, svm_sweeps, tree_sweeps


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # Azerbaijani text on Windows consoles

    # 1. data -------------------------------------------------------------
    _section("1. Load + clean")
    df = data_prep.clean(data_prep.load_raw())
    for i, line in enumerate(data_prep.CLEANING_LOG, 1):
        print(f"  [{i:2d}] {line}")

    # frozen, stratified 70/15/15 split (provisional median bin on log-price)
    y_log = np.log(df["price"].to_numpy(dtype=float))
    idx_tr, idx_val, idx_te = data_prep.split_indices(y_log)
    data_prep.freeze_split(idx_tr, idx_val, idx_te)
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[idx_tr] = True

    # features: category vocabularies learned on TRAIN rows only
    X, y, feat_names = data_prep.make_features(df, fit_mask)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_prep.train_val_test_split(X, y)
    assert np.array_equal(y_tr, y[idx_tr]), "split must match the frozen indices"

    # TRAIN-only statistics: imputation medians, tier threshold, scaling
    X_tr, X_val, X_te = data_prep.impute_median(X_tr, X_val, X_te)
    price = df["price"].to_numpy(dtype=float)
    price_tr, price_val, price_te = price[idx_tr], price[idx_val], price[idx_te]
    t_tr, tier_threshold = data_prep.make_tier_label(price_tr)
    t_val, _ = data_prep.make_tier_label(price_val, tier_threshold)
    t_te, _ = data_prep.make_tier_label(price_te, tier_threshold)
    Xs_tr, Xs_val, Xs_te = data_prep.standardize(X_tr, X_val, X_te)

    _section("2. Framing + split")
    print("  Task A (regression):     target log(price); metrics RMSE / MAE / R² "
          "(log and AZN scale)")
    print("  Task B (classification): premium = price > train median; metrics "
          "precision / recall / F1, ROC-AUC, confusion matrix")
    print(f"  features: {X.shape[1]} columns "
          f"({sum('=' in n for n in feat_names)} one-hot, "
          f"{sum(n.endswith('_missing') for n in feat_names)} missing-indicators)")
    n = len(df)
    for name, Xi, ti in (("train", X_tr, t_tr), ("val", X_val, t_val), ("test", X_te, t_te)):
        print(f"  {name:5s}: {len(Xi):6,d} rows ({len(Xi) / n:5.1%})  "
              f"premium share = {ti.mean():.3f}")
    print(f"  tier threshold (train median price): {tier_threshold:,.0f} AZN")

    _section("3. EDA")
    print("  " + eda.summary(df).replace("\n", "\n  "))
    paths = eda.run_eda(df, {"train": t_tr, "val": t_val, "test": t_te}, tier_threshold,
                        seed=data_prep.SEED)
    for p in paths:
        print(f"  saved {p}")

    # the experiment scripts reload the data through the same frozen split
    _section("4. Decision tree sweeps (validation)")
    tree_sweeps.main()

    _section("5. SVM sweeps (validation)")
    svm_sweeps.main()

    _section("6. SVM analysis (validation)")
    svm_analysis.main()

    _section("7. Benchmark vs scikit-learn (CV on train, one test evaluation)")
    benchmark.main()

    _section("8. Final results")
    print_results(benchmark.OUT_DIR)


def print_results(table_dir: str) -> None:
    """Print the headline tables written by benchmark.main()."""
    cv = pd.read_csv(os.path.join(table_dir, "benchmark_cv.csv"))
    print("5-fold CV on train (mean ± std; fit/predict seconds per fold):")
    for r in cv.itertuples():
        print(f"  {r.task}  {r.model:28s} {r.metric} = {r.cv_mean:.4f} ± {r.cv_std:.4f}"
              f"   fit {r.fit_s:7.2f}s   predict {r.predict_s:6.3f}s")

    test = pd.read_csv(os.path.join(table_dir, "benchmark_test.csv")).set_index("model")
    reg = test[test["rmse_log"].notna()]
    cls = test[test["f1"].notna()]
    print("\nTask A, test set (log price; AZN with smearing correction):")
    ev.compare({m: r[["rmse_log", "mae_log", "r2_log", "rmse_azn", "mae_azn"]].to_dict()
                for m, r in reg.iterrows()})
    print("\nTask B, test set (premium = positive class):")
    ev.compare({m: r[["precision", "recall", "f1", "auc"]].to_dict()
                for m, r in cls.iterrows()})


if __name__ == "__main__":
    main()
