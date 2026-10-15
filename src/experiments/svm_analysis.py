"""svm_analysis.py: Dev 3. SVM weights, hyperparameter effects and the gap to scikit-learn.

Run from repo root:  python -m src.experiments.svm_analysis

  1. weights      (Problem 6)  : which standardized features push a listing to "premium"
  2. hyperparams  (Problem 3b) : how the epoch count and learning-rate schedule move the
                                 margin and the objective (lambda is swept in svm_sweeps.py)
  3. sklearn gap  (Problem 5b) : ours vs SGDClassifier(hinge) and LinearSVC(hinge) with the
                                 matched lambda <-> alpha <-> C, on the same scaled features

VALIDATION only. The test set is never touched here (the single test evaluation is in
benchmark.py).
"""
from __future__ import annotations
import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from src import data_prep
from src import evaluate as ev
from src.svm import PegasosSVM
from src.experiments.tree_sweeps import FIG_DIR, load_splits, save_csv

SEED = data_prep.SEED
LAMBDA = 1e-3                      # best linear lambda by validation F1 in svm_sweeps.py
TOP_K = 12                         # weights shown per sign
SCHEDULES = {"pegasos": None, "invsqrt": 0.5, "constant": 0.01}   # schedule -> eta0
EPOCHS = [1, 2, 5, 10, 20, 50]


def objective(w, b, X, y_pm1, lam=LAMBDA) -> float:
    """Textbook J(w, b) = lambda/2 ||w||^2 + mean hinge (bias NOT regularised), for any model."""
    return float(0.5 * lam * w @ w + np.mean(np.maximum(0.0, 1.0 - y_pm1 * (X @ w + b))))


def val_scores(model, X, y):
    pred = model.predict(X)
    _, _, f1 = ev.precision_recall_f1(y, pred)
    return f1, ev.roc_auc(y, model.decision_function(X)), pred


# --- 1. weights ---------------------------------------------------------------
def weights(model, names):
    order = np.argsort(model.w)
    rows = [dict(rank=r, feature=names[i], weight=float(model.w[i]))
            for r, i in enumerate(order[::-1], 1)]
    save_csv(rows, "svm_weights.csv")

    top = np.r_[order[:TOP_K], order[-TOP_K:]]
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.barh(range(len(top)), model.w[top],
            color=["tab:blue" if v < 0 else "tab:red" for v in model.w[top]])
    ax.set_yticks(range(len(top)), [names[i] for i in top], fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("weight on standardized feature  (+ -> premium)")
    ax.set_title(f"Pegasos SVM weights (lambda={LAMBDA:g}), top {TOP_K} each sign", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "svm_weights.png"), dpi=200)
    plt.close(fig)

    print(f"\nTop {TOP_K} premium weights:")
    for i in order[::-1][:TOP_K]:
        print(f"  {model.w[i]:+.3f}  {names[i]}")
    print(f"Top {TOP_K} standard weights:")
    for i in order[:TOP_K]:
        print(f"  {model.w[i]:+.3f}  {names[i]}")
    return rows


# --- 2. epochs and learning-rate schedule (Problem 3b) --------------------------
def hyperparams(Xtr, ytr, Xva, yva):
    """Raw last iterates per schedule (average=False), plus the averaged default for reference."""
    rows, curves = [], {}
    for sched, eta0 in SCHEDULES.items():
        kw = dict(lambda_=LAMBDA, lr_schedule=sched, average=False, random_state=SEED)
        if eta0 is not None:
            kw["eta0"] = eta0
        for ep in EPOCHS:
            m = PegasosSVM(n_epochs=ep, **kw).fit(Xtr, ytr)
            f1, auc, _ = val_scores(m, Xva, yva)
            rows.append(dict(schedule=sched, eta0=eta0 or "", average=False, n_epochs=ep,
                             val_f1=f1, val_auc=auc, norm_w=m.norm_w_, margin=m.margin_width_,
                             n_sv=m.n_support_, final_objective=m.loss_history_[-1]))
            print(f"{sched:8s} epochs={ep:2d} val_f1={f1:.3f} auc={auc:.3f} "
                  f"margin={m.margin_width_:.3f} sv={m.n_support_} J={m.loss_history_[-1]:.4f}")
        curves[sched] = m.loss_history_                 # from the longest run

    avg = PegasosSVM(lambda_=LAMBDA, random_state=SEED).fit(Xtr, ytr)   # default: 10 epochs, averaged
    f1, auc, _ = val_scores(avg, Xva, yva)
    rows.append(dict(schedule="pegasos", eta0="", average=True, n_epochs=avg.n_epochs,
                     val_f1=f1, val_auc=auc, norm_w=avg.norm_w_, margin=avg.margin_width_,
                     n_sv=avg.n_support_, final_objective=avg.loss_history_[-1]))
    print(f"pegasos  epochs={avg.n_epochs:2d} AVERAGED val_f1={f1:.3f} auc={auc:.3f} "
          f"margin={avg.margin_width_:.3f} sv={avg.n_support_} J={avg.loss_history_[-1]:.4f}")
    save_csv(rows, "svm_epochs_schedule.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.3))
    for sched, h in curves.items():
        a1.plot(range(1, len(h) + 1), h, marker="o", ms=2, label=sched)
        r = [x for x in rows if x["schedule"] == sched and not x["average"]]
        a2.plot([x["n_epochs"] for x in r], [x["margin"] for x in r], marker="o", ms=3, label=sched)
    a1.plot([avg.n_epochs], [avg.loss_history_[-1]], "k*", ms=9, label="pegasos, averaged")
    a1.set_xlabel("epoch"); a1.set_ylabel("objective J(w, b)"); a1.set_yscale("log")
    a2.set_xlabel("epochs"); a2.set_ylabel("margin 2/||w||")
    for a in (a1, a2):
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle(f"Learning-rate schedule and epochs (lambda={LAMBDA:g})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "svm_epochs_schedule.png"), dpi=200)
    plt.close(fig)
    return rows


# --- 3. gap to scikit-learn (Problem 5b) ----------------------------------------
def sklearn_gap(Xtr, ytr, Xva, yva):
    n = len(Xtr)
    y_pm1 = np.where(ytr == 1, 1.0, -1.0)
    models = {
        # same objective as ours: alpha/2 ||w||^2 + mean hinge, so alpha = lambda
        "sklearn SGDClassifier(hinge)": SGDClassifier(loss="hinge", alpha=LAMBDA, random_state=SEED),
        # C * sum hinge + 1/2 ||w||^2  ==  (1 / (C n)) * [lambda/2 ||w||^2 + mean hinge] with C = 1/(lambda n)
        "sklearn LinearSVC(hinge)": LinearSVC(loss="hinge", C=1.0 / (LAMBDA * n), dual=True,
                                              max_iter=50_000, random_state=SEED),
        "ours PegasosSVM (last iterate)": PegasosSVM(lambda_=LAMBDA, average=False, random_state=SEED),
        "ours PegasosSVM (batch 256)": PegasosSVM(lambda_=LAMBDA, batch_size=256, random_state=SEED),
        "ours PegasosSVM": PegasosSVM(lambda_=LAMBDA, random_state=SEED),     # default: averaged
    }
    fitted, rows = {}, []
    for name, m in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            m.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
        converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
        w = np.ravel(m.coef_) if hasattr(m, "coef_") else m.w
        b = float(np.ravel(m.intercept_)[0]) if hasattr(m, "intercept_") else m.b
        fitted[name] = (m, w)
        f1, auc, pred = val_scores(m, Xva, yva)
        rows.append(dict(model=name, fit_seconds=fit_s, converged=converged, val_f1=f1, val_auc=auc,
                         train_objective=objective(w, b, Xtr, y_pm1),
                         norm_w=float(np.linalg.norm(w)), margin=2.0 / float(np.linalg.norm(w)),
                         n_sv=int(np.sum(y_pm1 * (Xtr @ w + b) <= 1.0)), bias=b))

    ours_m, ours_w = fitted["ours PegasosSVM"]
    ours_pred = ours_m.predict(Xva)
    for r in rows:
        m, w = fitted[r["model"]]
        r["cos_w_vs_ours"] = float(w @ ours_w / (np.linalg.norm(w) * np.linalg.norm(ours_w)))
        r["val_agreement_with_ours"] = float(np.mean(m.predict(Xva) == ours_pred))
    save_csv(rows, "svm_vs_sklearn.csv")

    print(f"\n{'model':32s} {'fit_s':>6} {'conv':>5} {'val_f1':>7} {'auc':>6} {'J_train':>8} "
          f"{'||w||':>7} {'margin':>7} {'n_sv':>6} {'cos_w':>6} {'agree':>6}")
    for r in rows:
        print(f"{r['model']:32s} {r['fit_seconds']:6.2f} {str(r['converged']):>5} {r['val_f1']:7.4f} "
              f"{r['val_auc']:6.4f} {r['train_objective']:8.4f} {r['norm_w']:7.3f} {r['margin']:7.3f} "
              f"{r['n_sv']:6d} {r['cos_w_vs_ours']:6.3f} {r['val_agreement_with_ours']:6.3f}")
    return rows


def main():
    d = load_splits()
    Xtr, Xva = data_prep.standardize(d["Xtr"], d["Xva"])     # TRAIN statistics only
    ytr, yva = d["ytr_cls"], d["yva_cls"]

    final = PegasosSVM(lambda_=LAMBDA, random_state=SEED).fit(Xtr, ytr)
    weights(final, d["names"])
    print()
    hyperparams(Xtr, ytr, Xva, yva)
    sklearn_gap(Xtr, ytr, Xva, yva)


if __name__ == "__main__":
    main()

