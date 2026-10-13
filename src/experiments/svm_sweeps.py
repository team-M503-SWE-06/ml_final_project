"""svm_sweeps.py: Dev 4. lambda and RFF-bandwidth sweeps for OUR Pegasos SVM.

Run from repo root:  python -m src.experiments.svm_sweeps
Selection uses VALIDATION only. The test set is never touched here.
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
from src.svm import PegasosSVM
try:
    from src.svm import RFFPegasosSVM
except ImportError:         
    RFFPegasosSVM = None
from src import evaluate as ev
from src.experiments.tree_sweeps import load_splits, save_csv

FIG_DIR = os.path.join("report", "figures")
SEED = data_prep.SEED
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
GAMMAS = [0.003, 0.01, 0.03, 0.1]      # RFF bandwidth, small because 72 features
N_COMPONENTS = 500


def standardized(d):
    """Scale with TRAIN statistics only (SVM is scale-sensitive)."""
    Xtr, Xva = data_prep.standardize(d["Xtr"], d["Xva"])
    return Xtr, Xva


def scores(model, X, y):
    pred = model.predict(X)
    _, _, f1 = ev.precision_recall_f1(y, pred)
    auc = ev.roc_auc(y, model.decision_function(X))
    return float(np.mean(pred == y)), f1, auc


def sweep_lambda(Xtr, Xva, ytr, yva):
    rows = []
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        m = PegasosSVM(lambda_=lam, random_state=SEED).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        _, ftr, _ = scores(m, Xtr, ytr)
        acc, fva, auc = scores(m, Xva, yva)
        n_sv = int(np.sum(m.support_vector_mask(Xtr, ytr)))
        rows.append(dict(model="linear", lambda_=lam, gamma="", train_f1=ftr, val_f1=fva,
                         val_acc=acc, val_auc=auc, norm_w=float(m.norm_w_),
                         margin=float(m.margin_width_), n_sv=n_sv,
                         final_loss=float(m.loss_history_[-1]), fit_seconds=fit_s))
        print(f"linear lam={lam:g} val_f1={fva:.3f} auc={auc:.3f} "
              f"margin={m.margin_width_:.3f} sv={n_sv} ({fit_s:.1f}s)")
    return rows


def sweep_rff(Xtr, Xva, ytr, yva, lam):
    rows = []
    for g in GAMMAS:
        t0 = time.perf_counter()
        m = RFFPegasosSVM(n_components=N_COMPONENTS, gamma=g, random_state=SEED,
                          lambda_=lam).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        _, ftr, _ = scores(m, Xtr, ytr)
        acc, fva, auc = scores(m, Xva, yva)
        rows.append(dict(model="rff", lambda_=lam, gamma=g, train_f1=ftr, val_f1=fva,
                         val_acc=acc, val_auc=auc, margin=float(m.margin_width_),
                         final_loss=float(m.loss_history_[-1]), fit_seconds=fit_s))
        print(f"rff gamma={g:g} lam={lam:g} val_f1={fva:.3f} auc={auc:.3f} ({fit_s:.1f}s)")
    return rows


def plot_lambda(rows):
    os.makedirs(FIG_DIR, exist_ok=True)
    lam = [r["lambda_"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogx(lam, [r["train_f1"] for r in rows], "o-", ms=3, label="train F1")
    ax.semilogx(lam, [r["val_f1"] for r in rows], "o-", ms=3, label="val F1")
    ax.set_xlabel("lambda"); ax.set_ylabel("F1 (premium)"); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.semilogx(lam, [r["margin"] for r in rows], "k--", label="margin 2/||w||")
    ax2.set_ylabel("margin width")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.set_title("Pegasos SVM: lambda sweep")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "svm_lambda_sweep.png"), dpi=200)
    plt.close(fig)


def plot_loss(Xtr, ytr, lam):
    """Confirms the training objective converges (Problem 4b)."""
    m = PegasosSVM(lambda_=lam, random_state=SEED).fit(Xtr, ytr)
    plt.figure(figsize=(5, 3.5))
    plt.plot(m.loss_history_, marker="o", ms=3)
    plt.xlabel("epoch"); plt.ylabel("objective J(w,b)")
    plt.title(f"Pegasos objective (lambda={lam:g})"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "svm_objective.png"), dpi=200)
    plt.close()


def main():
    d = load_splits()
    Xtr, Xva = standardized(d)
    ytr, yva = d["ytr_cls"], d["yva_cls"]

    lin = sweep_lambda(Xtr, Xva, ytr, yva)
    save_csv(lin, "svm_lambda_sweep.csv")
    plot_lambda(lin)

    best = max(lin, key=lambda r: r["val_f1"])          # VALIDATION only
    print("\nBEST linear (val F1): lambda =", best["lambda_"], round(best["val_f1"], 4))
    plot_loss(Xtr, ytr, best["lambda_"])
    if RFFPegasosSVM is None:
        print("RFFPegasosSVM not found in src/svm.py: skipping the RFF sweep.")
        return
    rff = sweep_rff(Xtr, Xva, ytr, yva, best["lambda_"])
    save_csv(rff, "svm_rff_sweep.csv")
    best_r = max(rff, key=lambda r: r["val_f1"])
    print("BEST rff (val F1): gamma =", best_r["gamma"], round(best_r["val_f1"], 4))


if __name__ == "__main__":
    main()