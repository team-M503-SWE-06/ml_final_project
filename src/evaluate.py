"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    raise NotImplementedError("TODO: sqrt(mean((y_true - y_pred)^2))")


def mae(y_true, y_pred) -> float:
    raise NotImplementedError("TODO: mean(|y_true - y_pred|)")


def r2(y_true, y_pred) -> float:
    raise NotImplementedError("TODO: 1 - SS_res/SS_tot")


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    raise NotImplementedError("TODO: confusion matrix")


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    raise NotImplementedError("TODO: precision/recall/f1")


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    raise NotImplementedError("TODO: ROC AUC")


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    raise NotImplementedError("TODO: print comparison table")
