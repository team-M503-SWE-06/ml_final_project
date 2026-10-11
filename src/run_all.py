"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator should:
  1. load + clean the data,
  2. build features, make the splits, standardize (for the SVM),
  3. train YOUR decision tree (Task A regression, Task B classification),
  4. train YOUR Pegasos SVM (Task B classification),
  5. train the scikit-learn baselines with matched settings,
  6. compute metrics and SAVE all figures to report/figures/,
  7. print a final comparison table.

Everything must be deterministic (fixed seeds). Keep print output tidy so a
grader can read the headline numbers straight from the console.
"""

from __future__ import annotations

import sys

import numpy as np

from . import data_prep, eda
# from .decision_tree import DecisionTree
# from .svm import PegasosSVM
# from . import evaluate


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

    # 2. Task A: regression (price) --------------------------------------
    #    - your DecisionTree(task="regression", criterion="mse")
    #    - sklearn DecisionTreeRegressor baseline

    # 3. Task B: classification (price tier) -----------------------------
    #    - make_tier_label using TRAIN median
    #    - your DecisionTree(task="classification")
    #    - your PegasosSVM (standardized features)
    #    - sklearn baselines (DecisionTreeClassifier, SVC/LinearSVC)

    # 4. metrics + figures + comparison ----------------------------------
    #    evaluate.compare({...}); save plots to report/figures/


if __name__ == "__main__":
    main()
