"""ensembles.py: Dev 2. Pruning, best-first growth, L1/L2 leaf regularisation,
our Random Forest and gradient boosting (Problem 9), and permutation importance (Problem 6).

Run from repo root:  python -m src.experiments.ensembles

Every choice (alpha, number of leaves, lambda, which ensemble variant) is made
on VALIDATION data. The test set is used once at the end, for the Problem 9
curves "test error as learners are added" of the already chosen models.
"""
from __future__ import annotations

import copy
import csv
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as SkGradientBoosting
from sklearn.ensemble import HistGradientBoostingRegressor as SkHistGradientBoosting
from sklearn.ensemble import RandomForestClassifier as SkForestClassifier
from sklearn.ensemble import RandomForestRegressor as SkForestRegressor

from src import evaluate as ev
from src.decision_tree import DecisionTree
from src.ensemble import GradientBoostingRegressor, RandomForest
from src.experiments import benchmark
from src.experiments.tree_sweeps import load_splits
from src.importance import one_hot_groups, permutation_importance

FIG_DIR = os.path.join("report", "figures")
OUT_DIR = os.path.join("report", "tables")
SEED = benchmark.SEED
N_TREES = 50                     # forest size; the curves show where error flattens
FOREST_REG_FEATURES = 0.33       # regression forests usually look at ~d/3 features
GB_PARAMS = dict(n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8)
GB_VARIANTS = {"no penalty": {}, "leaf_l2=10": dict(leaf_l2=10.0), "leaf_l1=3": dict(leaf_l1=3.0)}


def f1(y, pred):
    return ev.precision_recall_f1(y, pred)[2]


def save_csv(rows, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)


# --- single-tree regularisation ------------------------------------------------
def pruning(d):
    """Grow each tree once, prune copies along the cost-complexity path, score on validation."""
    rows, best = [], {}
    for task, crit, ytr, yva in (("cls", "gini", d["ytr_cls"], d["yva_cls"]),
                                 ("reg", "mse", d["ytr_reg"], d["yva_reg"])):
        full = DecisionTree(task="classification" if task == "cls" else "regression",
                            criterion=crit).fit(d["Xtr"], ytr)
        path = []
        copy.deepcopy(full)._prune(np.inf, path)
        alphas = np.unique([a for a, _ in path if a > 0])
        grid = np.unique(np.quantile(alphas, np.linspace(0.0, 0.995, 25)))
        for alpha in np.concatenate([[0.0], grid]):
            tree = copy.deepcopy(full).prune(float(alpha))
            score = f1(yva, tree.predict(d["Xva"])) if task == "cls" else ev.rmse(yva, tree.predict(d["Xva"]))
            rows.append(dict(task=task, ccp_alpha=float(alpha), leaves=tree.get_n_leaves(),
                             depth=tree.get_depth(), val_score=score))
        task_rows = [r for r in rows if r["task"] == task]
        best[task] = (max if task == "cls" else min)(task_rows, key=lambda r: r["val_score"])
        print(f"pruning {task}: full tree {full.get_n_leaves()} leaves -> best alpha "
              f"{best[task]['ccp_alpha']:.2e}, {best[task]['leaves']} leaves, "
              f"val {'F1' if task == 'cls' else 'RMSE'} {best[task]['val_score']:.4f}")
    save_csv(rows, "pruning_path.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, task, label in zip(axes, ("cls", "reg"), ("val F1 (tier)", "val RMSE (log price)")):
        r = [x for x in rows if x["task"] == task and x["ccp_alpha"] > 0]
        ax.semilogx([x["ccp_alpha"] for x in r], [x["val_score"] for x in r], marker="o", ms=3)
        ax.set_xlabel("ccp_alpha")
        ax.set_ylabel(label)
        ax.set_title(f"Cost-complexity pruning ({task})")
    save_fig(fig, "pruning_alpha.png")
    return best


def best_first(d):
    rows = []
    for k in (50, 100, 200, 500, 1000, 2000, 4000):
        cls = DecisionTree(criterion="gini", max_leaf_nodes=k).fit(d["Xtr"], d["ytr_cls"])
        reg = DecisionTree(task="regression", criterion="mse", max_leaf_nodes=k).fit(d["Xtr"], d["ytr_reg"])
        rows.append(dict(max_leaf_nodes=k, val_f1=f1(d["yva_cls"], cls.predict(d["Xva"])),
                         cls_depth=cls.get_depth(),
                         val_rmse=ev.rmse(d["yva_reg"], reg.predict(d["Xva"])), reg_depth=reg.get_depth()))
        print(f"best-first {k:5d} leaves: val F1 {rows[-1]['val_f1']:.4f} (depth {cls.get_depth()}), "
              f"val RMSE {rows[-1]['val_rmse']:.4f} (depth {reg.get_depth()})")
    save_csv(rows, "best_first.csv")
    return rows


def leaf_regularisation(d):
    """L2 / L1 on leaf values of single trees (the split search is unchanged)."""
    rows = []
    for l2 in (0.0, 1.0, 3.0, 10.0, 30.0):
        m = DecisionTree(**benchmark.TREE_CLS_PARAMS, leaf_l2=l2).fit(d["Xtr"], d["ytr_cls"])
        rows.append(dict(task="cls", leaf_l2=l2, leaf_l1=0.0, val_f1=f1(d["yva_cls"], m.predict(d["Xva"])),
                         val_auc=ev.roc_auc(d["yva_cls"], m.predict_proba(d["Xva"])[:, 1])))
    for l2, l1 in ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (10.0, 0.0), (30.0, 0.0),
                   (0.0, 1.0), (0.0, 3.0), (0.0, 10.0)):
        m = DecisionTree(task="regression", criterion="mse", **benchmark.TREE_REG_PARAMS,
                         leaf_l2=l2, leaf_l1=l1).fit(d["Xtr"], d["ytr_reg"])
        rows.append(dict(task="reg", leaf_l2=l2, leaf_l1=l1, val_rmse=ev.rmse(d["yva_reg"], m.predict(d["Xva"]))))
    for r in rows:
        print("leaf reg", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    save_csv(rows, "leaf_regularisation.csv")
    return rows


# --- ensembles -----------------------------------------------------------------
def random_forests(d, test):
    """Error vs number of trees (validation), OOB, and sklearn with the same settings."""
    out, rows = {}, []
    specs = (("cls", dict(task="classification", max_features="sqrt"), d["ytr_cls"], d["yva_cls"], test["c_te"]),
             ("reg", dict(task="regression", max_features=FOREST_REG_FEATURES), d["ytr_reg"], d["yva_reg"],
              test["y_te"]))
    for task, params, ytr, yva, yte in specs:
        t0 = time.perf_counter()
        forest = RandomForest(**params, n_estimators=N_TREES, oob_score=True, n_jobs=-1,
                              random_state=SEED).fit(d["Xtr"], ytr)
        fit_s = time.perf_counter() - t0
        score = f1 if task == "cls" else ev.rmse
        val_curve = [score(yva, p) for p in forest.staged_predict(d["Xva"])]
        test_curve = [score(yte, p) for p in forest.staged_predict(test["X_te"])]
        sk_class = SkForestClassifier if task == "cls" else SkForestRegressor
        t0 = time.perf_counter()
        sk = sk_class(n_estimators=N_TREES, max_features=params["max_features"], random_state=SEED,
                      n_jobs=-1).fit(d["Xtr"], ytr)
        sk_fit = time.perf_counter() - t0
        single = DecisionTree(task=params["task"], criterion="gini" if task == "cls" else "mse").fit(d["Xtr"], ytr)
        extra = {}
        if task == "cls":
            extra = dict(val_auc=ev.roc_auc(yva, forest.predict_proba(d["Xva"])[:, 1]),
                         sk_val_auc=ev.roc_auc(yva, sk.predict_proba(d["Xva"])[:, 1]))
        out[task] = dict(val_curve=val_curve, test_curve=test_curve)
        rows.append(dict(task=task, n_trees=N_TREES, val_score=val_curve[-1], test_score=test_curve[-1],
                         oob_score=forest.oob_score_, single_tree_val=score(yva, single.predict(d["Xva"])),
                         sk_val_score=score(yva, sk.predict(d["Xva"])), fit_s=fit_s, sk_fit_s=sk_fit, **extra))
        print(f"forest {task}: val {val_curve[0]:.4f} (1 tree) -> {val_curve[-1]:.4f} ({N_TREES} trees), "
              f"sklearn {rows[-1]['sk_val_score']:.4f}, single full tree {rows[-1]['single_tree_val']:.4f}, "
              f"OOB {forest.oob_score_:.4f}, fit {fit_s:.0f}s (sklearn {sk_fit:.1f}s)")
    save_csv(rows, "forest.csv")
    return out, rows


def _fit_boosting(job):
    name, extra, X, y = job
    t0 = time.perf_counter()
    model = GradientBoostingRegressor(**GB_PARAMS, **extra, random_state=SEED).fit(X, y)
    return name, model, time.perf_counter() - t0


def boosting(d, test):
    """Our boosting with/without leaf penalties vs sklearn; the variant is chosen on validation."""
    jobs = [(name, extra, d["Xtr"], d["ytr_reg"]) for name, extra in GB_VARIANTS.items()]
    with Pool(len(jobs)) as pool:
        fitted = pool.map(_fit_boosting, jobs)
    rows, curves = [], {}
    for name, model, fit_s in fitted:
        curve = [ev.rmse(d["yva_reg"], p) for p in model.staged_predict(d["Xva"])]
        curves[name] = (model, curve)
        rows.append(dict(model=f"ours {name}", val_rmse=curve[-1], best_step=int(np.argmin(curve)) + 1,
                         best_val_rmse=min(curve), fit_s=fit_s))
    sk = SkGradientBoosting(n_estimators=GB_PARAMS["n_estimators"], learning_rate=GB_PARAMS["learning_rate"],
                            max_depth=GB_PARAMS["max_depth"], subsample=GB_PARAMS["subsample"],
                            random_state=SEED).fit(d["Xtr"], d["ytr_reg"])
    rows.append(dict(model="sklearn GradientBoosting", val_rmse=ev.rmse(d["yva_reg"], sk.predict(d["Xva"]))))
    for l2 in (0.0, 10.0):
        hist = SkHistGradientBoosting(max_iter=GB_PARAMS["n_estimators"], learning_rate=GB_PARAMS["learning_rate"],
                                      max_depth=GB_PARAMS["max_depth"], l2_regularization=l2,
                                      early_stopping=False, random_state=SEED).fit(d["Xtr"], d["ytr_reg"])
        rows.append(dict(model=f"sklearn HistGradientBoosting l2={l2:g}",
                         val_rmse=ev.rmse(d["yva_reg"], hist.predict(d["Xva"]))))
    chosen = min(curves, key=lambda k: curves[k][1][-1])
    model = curves[chosen][0]
    test_curve = [ev.rmse(test["y_te"], p) for p in model.staged_predict(test["X_te"])]
    rows.append(dict(model=f"ours {chosen} (chosen on val), TEST", test_rmse=test_curve[-1]))
    for r in rows:
        print("boosting", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    save_csv(rows, "boosting.csv")
    return curves, chosen, test_curve


def plot_ensemble_curves(forest_curves, gb_curves, gb_chosen, gb_test_curve):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    k = np.arange(1, N_TREES + 1)
    axes[0].plot(k, forest_curves["cls"]["val_curve"], label="validation")
    axes[0].plot(k, forest_curves["cls"]["test_curve"], label="test")
    axes[0].set(xlabel="number of trees", ylabel="F1 (price tier)", title="Our Random Forest, Task B")
    axes[1].plot(k, forest_curves["reg"]["val_curve"], label="validation")
    axes[1].plot(k, forest_curves["reg"]["test_curve"], label="test")
    axes[1].set(xlabel="number of trees", ylabel="RMSE (log price)", title="Our Random Forest, Task A")
    steps = np.arange(1, GB_PARAMS["n_estimators"] + 1)
    for name, (_, curve) in gb_curves.items():
        axes[2].plot(steps, curve, label=f"val, {name}")
    axes[2].plot(steps, gb_test_curve, "k--", label=f"test, {gb_chosen}")
    axes[2].set(xlabel="boosting steps", ylabel="RMSE (log price)", title="Our gradient boosting, Task A")
    axes[2].set_ylim(top=0.6)
    for ax in axes:
        ax.legend(fontsize=7)
    save_fig(fig, "ensemble_error_vs_learners.png")


# --- permutation importance ------------------------------------------------------
def importance(d):
    """MDI (training data) vs permutation importance (validation) for the final trees."""
    groups = one_hot_groups(d["names"])
    rows = []
    specs = (("reg", DecisionTree(task="regression", criterion="mse", **benchmark.TREE_REG_PARAMS),
              d["ytr_reg"], d["yva_reg"], ev.rmse, False, "predict"),
             ("cls", DecisionTree(task="classification", **benchmark.TREE_CLS_PARAMS),
              d["ytr_cls"], d["yva_cls"], ev.roc_auc, True, "proba"))
    for task, model, ytr, yva, metric, higher, response in specs:
        model.fit(d["Xtr"], ytr)
        perm = permutation_importance(model, d["Xva"], yva, metric, groups, higher_is_better=higher,
                                      response=response, seed=SEED)
        mdi = {g: float(model.feature_importances_[cols].sum()) for g, cols in groups.items()}
        for g, (mean, std) in perm.items():
            rows.append(dict(task=task, feature=g, permutation=mean, permutation_std=std, mdi=mdi[g]))
        top = list(perm)[:8]
        print(f"importance {task} (permutation on val | MDI on train):")
        for g in top:
            print(f"   {g:20s} {perm[g][0]:.4f} +- {perm[g][1]:.4f} | {mdi[g]:.3f}")
    save_csv(rows, "importance_mdi_vs_permutation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task, label in zip(axes, ("reg", "cls"), ("RMSE increase (Task A)", "AUC drop (Task B)")):
        r = sorted((x for x in rows if x["task"] == task), key=lambda x: -x["permutation"])[:10][::-1]
        y = np.arange(len(r))
        perm_share = np.array([x["permutation"] for x in r]) / max(sum(x["permutation"] for x in r), 1e-12)
        ax.barh(y + 0.2, perm_share, height=0.4, label=f"permutation ({label}), share")
        ax.barh(y - 0.2, [x["mdi"] for x in r], height=0.4, label="MDI (train)")
        ax.set_yticks(y, [x["feature"] for x in r])
        ax.legend(fontsize=7)
        ax.set_title(f"Feature importance, final tree ({task})")
    save_fig(fig, "importance_mdi_vs_permutation.png")
    return rows


def main():
    t_all = time.perf_counter()
    d = load_splits()
    Xtr_check, _, _, X_te, y_te, c_te, _, _ = benchmark.load_all()
    assert np.array_equal(Xtr_check, d["Xtr"]), "train matrix must match the sweeps"
    test = dict(X_te=X_te, y_te=y_te, c_te=c_te)

    print("\n-- cost-complexity pruning (validation) --")
    pruning(d)
    print("\n-- best-first growth (validation) --")
    best_first(d)
    print("\n-- L1/L2 leaf regularisation (validation) --")
    leaf_regularisation(d)
    print("\n-- Random Forest from our trees --")
    forest_curves, _ = random_forests(d, test)
    print("\n-- gradient boosting from our trees --")
    gb_curves, gb_chosen, gb_test_curve = boosting(d, test)
    plot_ensemble_curves(forest_curves, gb_curves, gb_chosen, gb_test_curve)
    print("\n-- permutation importance vs MDI (validation) --")
    importance(d)
    print(f"\nensembles done in {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
