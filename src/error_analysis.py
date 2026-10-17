"""
error_analysis.py: where do our decision trees make their biggest mistakes?

Uses the same cleaning, frozen split and features as run_all, trains one tree
per task on TRAIN rows, and breaks the TEST errors down by price band
(luxury listings), by district (sparse districts) and by distance to the tier threshold.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_prep
from .decision_tree import DecisionTree

FIG_DIR = "report/figures"
TABLE_DIR = "report/tables"

REG_PARAMS = {"criterion": "mse", "min_samples_split": 75}
CLS_PARAMS = {"criterion": "gini", "max_depth": 20, "min_impurity_decrease": 2e-5}
BAND_LABELS = ["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 luxury"]


def prepare():
    df = data_prep.clean(data_prep.load_raw())
    y_log = np.log(df["price"].to_numpy(dtype=float))
    idx_tr, idx_val, idx_te = data_prep.split_indices(y_log)
    data_prep.freeze_split(idx_tr, idx_val, idx_te)
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[idx_tr] = True
    X, y, _ = data_prep.make_features(df, fit_mask)
    X_tr, X_te = data_prep.impute_median(X[idx_tr], X[idx_te])
    return df, X_tr, X_te, y[idx_tr], y[idx_te], idx_tr, idx_te


def add_groups(df, idx_tr, idx_te):
    price_all = df["price"].to_numpy(dtype=float)
    price_tr = price_all[idx_tr]
    test = df.iloc[idx_te].reset_index(drop=True).copy()

    edges = np.quantile(price_tr, [0.2, 0.4, 0.6, 0.8])
    band = np.searchsorted(edges, test["price"].to_numpy(dtype=float), side="right")
    test["price_band"] = pd.Categorical(
        [BAND_LABELS[i] for i in band], categories=BAND_LABELS, ordered=True
    )
    test["luxury_top5"] = np.where(
        test["price"] >= np.quantile(price_tr, 0.95), "top 5% price", "other"
    )

    train_counts = df.iloc[idx_tr]["district"].value_counts()
    n_train = test["district"].map(train_counts).fillna(0)
    test["district_group"] = np.where(
        n_train >= data_prep.MIN_LEVEL_COUNT, "common district", "sparse or missing district"
    )
    return test, price_tr


def summarize_regression(test, col):
    g = test.groupby(col, observed=True)
    table = pd.DataFrame({
        "n": g.size(),
        "rmse_log": np.sqrt(g["sq_err"].mean()),
        "mae_log": g["abs_err"].mean(),
        "median_pct_error": (np.exp(g["abs_err"].median()) - 1) * 100,
    })
    return table.round(3)


def summarize_classification(test, col):
    g = test.groupby(col, observed=True)
    table = pd.DataFrame({"n": g.size(), "error_rate": 1 - g["correct"].mean()})
    return table.round(3)


def show(title, table, name):
    print(f"\n  {title}")
    print("  " + table.to_string().replace("\n", "\n  "))
    table.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"))


def save_figure(y_te, pred_log, band_table, district_table):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(y_te, pred_log, s=4, alpha=0.3)
    lims = [y_te.min(), y_te.max()]
    axes[0].plot(lims, lims, color="red")
    axes[0].set_xlabel("actual log(price)")
    axes[0].set_ylabel("predicted log(price)")
    axes[0].set_title("Predicted vs actual (test)")

    axes[1].bar(band_table.index.astype(str), band_table["rmse_log"])
    axes[1].set_ylabel("RMSE (log price)")
    axes[1].set_title("Error by price band")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(district_table.index.astype(str), district_table["rmse_log"])
    axes[2].set_ylabel("RMSE (log price)")
    axes[2].set_title("Error by district frequency")
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "error_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {path}")


def main() -> None:
    df, X_tr, X_te, y_tr, y_te, idx_tr, idx_te = prepare()
    test, price_tr = add_groups(df, idx_tr, idx_te)
    os.makedirs(TABLE_DIR, exist_ok=True)

    reg = DecisionTree(task="regression", **REG_PARAMS).fit(X_tr, y_tr)
    pred_log = reg.predict(X_te)
    test["abs_err"] = np.abs(pred_log - y_te)
    test["sq_err"] = (pred_log - y_te) ** 2
    test["pred_price"] = np.exp(pred_log)
    print(f"  Task A overall: RMSE(log) = {np.sqrt(test['sq_err'].mean()):.3f}, "
          f"MAE(log) = {test['abs_err'].mean():.3f}")

    band_table = summarize_regression(test, "price_band")
    show("Task A errors by price band", band_table, "error_reg_by_price_band")
    show("Task A errors, top 5% price vs rest",
         summarize_regression(test, "luxury_top5"), "error_reg_by_luxury")
    district_table = summarize_regression(test, "district_group")
    show("Task A errors by district frequency", district_table, "error_reg_by_district")

    worst = test.nlargest(10, "abs_err")[
        ["category", "district", "area", "rooms", "price", "pred_price", "abs_err"]
    ].round(2)
    show("Task A: 10 largest errors", worst, "error_reg_worst10")

    t_tr, threshold = data_prep.make_tier_label(price_tr)
    t_te, _ = data_prep.make_tier_label(test["price"].to_numpy(dtype=float), threshold)
    cls = DecisionTree(task="classification", **CLS_PARAMS).fit(X_tr, t_tr)
    test["correct"] = cls.predict(X_te) == t_te
    near = np.abs(np.log(test["price"]) - np.log(threshold)) < 0.1
    test["threshold_distance"] = np.where(near, "within 10% of threshold", "farther away")
    print(f"\n  Task B overall accuracy: {test['correct'].mean():.3f}")

    for col in ("price_band", "district_group", "threshold_distance"):
        show(f"Task B error rate by {col}",
             summarize_classification(test, col), f"error_cls_by_{col}")

    save_figure(y_te, pred_log, band_table, district_table)


if __name__ == "__main__":
    main()