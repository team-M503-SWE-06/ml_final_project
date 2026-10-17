"""tree_tuning.py: Dev 2. Evidence for the final tree configs used in benchmark.py.

Run from repo root:  python -m src.experiments.tree_tuning

tree_sweeps.py varies one hyperparameter at a time. The final configs also
use min_samples_split (Task A) and min_impurity_decrease (Task B), so this
script searches them jointly:
  1. a grid on VALIDATION (72 classification and 80 regression configs),
  2. 5-fold CV on TRAIN for the top 3 of each task.
The test set is never touched here.

benchmark.TREE_CLS_PARAMS / TREE_REG_PARAMS are then checked against this
evidence: each must be in the validation top 3 and within one CV std of the
best CV mean. For Task B the depth-20 tree is used instead of the unlimited
one: same validation F1 (difference below 1e-4), CV within one std, fewer
leaves and a bounded depth. For Task A min_samples_split=75 is best on both
validation and CV.
"""
from __future__ import annotations
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np

from src import evaluate as ev
from src.decision_tree import DecisionTree
from src.experiments import benchmark
from src.experiments.tree_sweeps import load_splits, save_csv

SEED = benchmark.SEED
N_FOLDS = benchmark.N_FOLDS
N_TOP = 3
N_JOBS = min(5, os.cpu_count() or 1)

CLS_GRID = [dict(criterion=c, max_depth=d, min_impurity_decrease=mid, min_samples_leaf=leaf)
            for c, d, mid, leaf in itertools.product(
                ("gini", "entropy"), (15, 20, None), (0.0, 5e-6, 1e-5, 2e-5), (1, 3, 5))]
REG_GRID = [dict(max_depth=d, min_samples_split=mss, min_impurity_decrease=mid)
            for d, mss, mid in itertools.product(
                (12, 15, 20, None), (50, 75, 100, 150, 200), (0.0, 1e-5, 5e-5, 1e-4))]

_DATA = None


def _init_worker():
    global _DATA
    _DATA = load_splits()


def _make_tree(task, params):
    if task == "cls":
        return DecisionTree(task="classification", **params)
    return DecisionTree(task="regression", criterion="mse", **params)


def _validation_row(job):
    task, params = job
    d = _DATA
    t0 = time.perf_counter()
    if task == "cls":
        m = _make_tree(task, params).fit(d["Xtr"], d["ytr_cls"])
        fit_s = time.perf_counter() - t0
        row = dict(params, train_f1=ev.precision_recall_f1(d["ytr_cls"], m.predict(d["Xtr"]))[2],
                   val_f1=ev.precision_recall_f1(d["yva_cls"], m.predict(d["Xva"]))[2],
                   val_auc=ev.roc_auc(d["yva_cls"], m.predict_proba(d["Xva"])[:, 1]))
    else:
        m = _make_tree(task, params).fit(d["Xtr"], d["ytr_reg"])
        fit_s = time.perf_counter() - t0
        p_va = m.predict(d["Xva"])
        row = dict(params, train_rmse=ev.rmse(d["ytr_reg"], m.predict(d["Xtr"])),
                   val_rmse=ev.rmse(d["yva_reg"], p_va), val_mae=ev.mae(d["yva_reg"], p_va),
                   val_r2=ev.r2(d["yva_reg"], p_va))
    row.update(leaves=m.get_n_leaves(), depth=m.get_depth(), fit_s=round(fit_s, 2))
    return row


def _cv_row(job):
    task, params = job
    d = _DATA
    y = d["ytr_cls"] if task == "cls" else d["ytr_reg"]
    folds = benchmark.kfold_indices(len(y), N_FOLDS, SEED)
    scores = []
    for i in range(N_FOLDS):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != i])
        pred = _make_tree(task, params).fit(d["Xtr"][tr], y[tr]).predict(d["Xtr"][va])
        if task == "cls":
            scores.append(ev.precision_recall_f1(y[va], pred)[2])
        else:
            scores.append(ev.rmse(y[va], pred))
    return dict(task=task, **params, cv_mean=float(np.mean(scores)), cv_std=float(np.std(scores)))


def _params(row, keys):
    return {k: row[k] for k in keys}


def check_final(task, cv_rows, top_rows, final, keys, higher_is_better):
    """Print where the benchmark config sits in the validation top N and the CV ranking."""
    in_top = any(_params(r, keys) == final for r in top_rows)
    best = (max if higher_is_better else min)(cv_rows, key=lambda r: r["cv_mean"])
    mine = next((r for r in cv_rows if _params(r, keys) == final), None)
    print(f"\n{task}: benchmark config {final}")
    print(f"  in validation top {N_TOP}: {in_top}")
    if mine is None:
        print("  not in the CV candidates")
        return False
    gap = abs(mine["cv_mean"] - best["cv_mean"])
    ok = in_top and gap <= best["cv_std"]
    print(f"  CV {mine['cv_mean']:.4f} +/- {mine['cv_std']:.4f}; best CV {best['cv_mean']:.4f} "
          f"+/- {best['cv_std']:.4f} ({_params(best, keys)}); gap {gap:.4f} -> "
          f"{'within one std' if gap <= best['cv_std'] else 'NOT within one std'}")
    return ok


def main():
    t_all = time.perf_counter()
    cls_keys = ("criterion", "max_depth", "min_impurity_decrease", "min_samples_leaf")
    reg_keys = ("max_depth", "min_samples_split", "min_impurity_decrease")
    print(f"grid: {len(CLS_GRID)} classification + {len(REG_GRID)} regression configs, "
          f"{N_JOBS} processes")

    with Pool(N_JOBS, initializer=_init_worker) as pool:
        cls_rows = pool.map(_validation_row, [("cls", p) for p in CLS_GRID])
        reg_rows = pool.map(_validation_row, [("reg", p) for p in REG_GRID])
        cls_rows.sort(key=lambda r: -r["val_f1"])
        reg_rows.sort(key=lambda r: r["val_rmse"])
        save_csv(cls_rows, "tree_tuning_cls.csv")
        save_csv(reg_rows, "tree_tuning_reg.csv")

        cls_top, reg_top = cls_rows[:N_TOP], reg_rows[:N_TOP]
        jobs = [("cls", _params(r, cls_keys)) for r in cls_top]
        jobs += [("reg", _params(r, reg_keys)) for r in reg_top]
        cv_rows = pool.map(_cv_row, jobs)
        save_csv(cv_rows, "tree_tuning_cv.csv")

    print(f"\nTOP {N_TOP} classification (validation F1):")
    for r in cls_top:
        print(f"  {_params(r, cls_keys)}  F1 {r['val_f1']:.4f}  AUC {r['val_auc']:.4f}  "
              f"leaves {r['leaves']}")
    print(f"TOP {N_TOP} regression (validation RMSE):")
    for r in reg_top:
        print(f"  {_params(r, reg_keys)}  RMSE {r['val_rmse']:.4f}  R2 {r['val_r2']:.4f}  "
              f"leaves {r['leaves']}")
    print(f"\n{N_FOLDS}-fold CV on train:")
    for r in cv_rows:
        print(f"  {r['task']}  {_params(r, cls_keys if r['task'] == 'cls' else reg_keys)}  "
              f"{r['cv_mean']:.4f} +/- {r['cv_std']:.4f}")

    final_cls = dict(criterion="gini", max_depth=None, min_impurity_decrease=0.0,
                     min_samples_leaf=1)
    final_cls.update(benchmark.TREE_CLS_PARAMS)
    final_reg = dict(max_depth=None, min_samples_split=2, min_impurity_decrease=0.0)
    final_reg.update(benchmark.TREE_REG_PARAMS)
    ok_cls = check_final("Task B", [r for r in cv_rows if r["task"] == "cls"], cls_top,
                         final_cls, cls_keys, higher_is_better=True)
    ok_reg = check_final("Task A", [r for r in cv_rows if r["task"] == "reg"], reg_top,
                         final_reg, reg_keys, higher_is_better=False)
    print(f"\nfinal configs supported by the evidence: {ok_cls and ok_reg}  "
          f"({time.perf_counter() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
