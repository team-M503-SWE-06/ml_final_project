"""evaluate.py: metrics + comparison helpers (NumPy only)."""
from __future__ import annotations
import numpy as np


# --- Regression (Task A: price) ---
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# --- Classification (Task B: price tier) ---
def confusion_matrix(y_true, y_pred):
    """[[TN, FP], [FN, TP]] with labels {0,1}."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f1)


def _rankdata(a):
    """Average ranks (1-based); ties get the mean rank."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    """AUC via the Mann-Whitney U formulation."""
    y_true = np.asarray(y_true)
    ranks = _rankdata(scores)
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


# --- Comparison helper ---
def compare(name_to_metrics: dict) -> None:
    """Print a table. name_to_metrics = {'model': {'metric': value, ...}}."""
    metrics = sorted({m for d in name_to_metrics.values() for m in d})
    w = max(len(n) for n in name_to_metrics) + 2
    print("model".ljust(w) + "".join(m.rjust(12) for m in metrics))
    for name, d in name_to_metrics.items():
        row = name.ljust(w)
        for m in metrics:
            v = d.get(m)
            row += f"{v:12.4f}" if isinstance(v, (int, float)) else "-".rjust(12)
        print(row)