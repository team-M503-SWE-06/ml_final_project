"""tree_sweeps.py: Dev 4. Depth / criterion / min_samples_leaf sweeps for OUR tree.

Run from repo root:  python -m src.experiments.tree_sweeps
Model selection uses VALIDATION only. The test set is never touched here.
"""
from __future__ import annotations
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data_prep
from src.decision_tree import DecisionTree
from src import evaluate as ev

FIG_DIR = os.path.join("report", "figures")
OUT_DIR = os.path.join("report", "tables")
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
LEAVES = [1, 5, 10, 25, 50, 100]
SEED = data_prep.SEED


def load_splits():
    """The ONLY place that touches data_prep. y from make_features is already log(price)."""
    df = data_prep.clean(data_prep.load_raw())
    y_all = np.log(df["price"].to_numpy(dtype=float))
    idx_tr, idx_va, idx_te = data_prep.split_indices(y_all, seed=SEED)

    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[idx_tr] = True                       # one-hot levels learned on TRAIN only
    X, y, names = data_prep.make_features(df, fit_mask=fit_mask)

    Xtr, Xva = X[idx_tr], X[idx_va]
    ytr, yva = y[idx_tr], y[idx_va]
    Xtr, Xva = data_prep.impute_median(Xtr, Xva)  # TRAIN medians; tree can't handle NaN

    tier_tr, thr = data_prep.make_tier_label(ytr)             # threshold from TRAIN only
    tier_va, _ = data_prep.make_tier_label(yva, threshold=thr)
    return {"Xtr": Xtr, "Xva": Xva, "ytr_reg": ytr, "yva_reg": yva,
            "ytr_cls": tier_tr, "yva_cls": tier_va, "names": names}


def _cls_scores(model, X, y):
    pred = model.predict(X)
    _, _, f1 = ev.precision_recall_f1(y, pred)
    return float(np.mean(pred == y)), f1


def sweep_classification(d):
    rows = []
    for crit in ("gini", "entropy"):
        for depth in DEPTHS:
            t0 = time.perf_counter()
            m = DecisionTree(task="classification", criterion=crit,
                             max_depth=depth, random_state=SEED).fit(d["Xtr"], d["ytr_cls"])
            fit_s = time.perf_counter() - t0
            atr, ftr = _cls_scores(m, d["Xtr"], d["ytr_cls"])
            ava, fva = _cls_scores(m, d["Xva"], d["yva_cls"])
            rows.append(dict(task="cls", criterion=crit, max_depth=depth, train_f1=ftr,
                             val_f1=fva, train_acc=atr, val_acc=ava, fit_seconds=fit_s))
            print(f"cls {crit:7s} depth={depth:2d} train_f1={ftr:.3f} val_f1={fva:.3f} ({fit_s:.1f}s)")
    return rows


def sweep_regression(d):
    rows = []
    for depth in DEPTHS:
        t0 = time.perf_counter()
        m = DecisionTree(task="regression", criterion="mse",
                         max_depth=depth, random_state=SEED).fit(d["Xtr"], d["ytr_reg"])
        fit_s = time.perf_counter() - t0
        rtr = ev.rmse(d["ytr_reg"], m.predict(d["Xtr"]))
        pva = m.predict(d["Xva"])
        rva, r2va = ev.rmse(d["yva_reg"], pva), ev.r2(d["yva_reg"], pva)
        rows.append(dict(task="reg", criterion="mse", max_depth=depth, train_rmse=rtr,
                         val_rmse=rva, val_r2=r2va, fit_seconds=fit_s))
        print(f"reg mse depth={depth:2d} train_rmse={rtr:.3f} val_rmse={rva:.3f} "
              f"val_r2={r2va:.3f} ({fit_s:.1f}s)")
    return rows


def sweep_min_leaf(d, crit, depth):
    """Second hyperparameter: min_samples_leaf, on a deliberately deep tree."""
    rows = []
    for leaf in LEAVES:
        m = DecisionTree(task="classification", criterion=crit, max_depth=depth,
                         min_samples_leaf=leaf, random_state=SEED).fit(d["Xtr"], d["ytr_cls"])
        _, ftr = _cls_scores(m, d["Xtr"], d["ytr_cls"])
        _, fva = _cls_scores(m, d["Xva"], d["yva_cls"])
        rows.append(dict(task="cls", criterion=crit, max_depth=depth,
                         min_samples_leaf=leaf, train_f1=ftr, val_f1=fva))
        print(f"cls leaf={leaf:3d} train_f1={ftr:.3f} val_f1={fva:.3f}")
    return rows


def save_csv(rows, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_curves(x, series, xlabel, ylabel, title, fname):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(5, 3.5))
    for label, ys in series.items():
        plt.plot(x, ys, marker="o", ms=3, label=label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close()


def main():
    d = load_splits()
    print("train", d["Xtr"].shape, "val", d["Xva"].shape,
          "| premium share train/val:", d["ytr_cls"].mean().round(3), d["yva_cls"].mean().round(3))

    cls = sweep_classification(d)
    reg = sweep_regression(d)
    save_csv(cls, "tree_cls_sweep.csv")
    save_csv(reg, "tree_reg_sweep.csv")

    for crit in ("gini", "entropy"):
        r = [x for x in cls if x["criterion"] == crit]
        plot_curves([x["max_depth"] for x in r],
                    {"train F1": [x["train_f1"] for x in r], "val F1": [x["val_f1"] for x in r]},
                    "max_depth", "F1 (premium)", f"Tree ({crit}): depth sweep",
                    f"tree_cls_depth_{crit}.png")
    plot_curves(DEPTHS,
                {"train RMSE": [x["train_rmse"] for x in reg],
                 "val RMSE": [x["val_rmse"] for x in reg]},
                "max_depth", "RMSE (log price)", "Tree (MSE): depth sweep", "tree_reg_depth.png")

    best_c = max(cls, key=lambda r: r["val_f1"])       # VALIDATION only
    best_r = min(reg, key=lambda r: r["val_rmse"])
    print("\nBEST cls (val F1):", best_c["criterion"], "depth", best_c["max_depth"],
          round(best_c["val_f1"], 4))
    print("BEST reg (val RMSE): depth", best_r["max_depth"], round(best_r["val_rmse"], 4))

    leaf = sweep_min_leaf(d, best_c["criterion"], max(DEPTHS))
    save_csv(leaf, "tree_min_leaf_sweep.csv")
    plot_curves(LEAVES,
                {"train F1": [x["train_f1"] for x in leaf], "val F1": [x["val_f1"] for x in leaf]},
                "min_samples_leaf", "F1 (premium)", "Tree: min_samples_leaf sweep",
                "tree_cls_min_leaf.png")


if __name__ == "__main__":
    main()