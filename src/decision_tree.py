"""
decision_tree.py: a CART-style decision tree built FROM SCRATCH (NumPy only).

One class covers both tasks of the project:
  - classification (criterion = "gini" or "entropy")  -> price tier (Task B)
  - regression     (criterion = "mse")                -> log(price) (Task A)

Every split is a binary threshold split on one numeric feature
(x[feature] <= threshold goes left), chosen to maximise the impurity decrease

    IG = I(parent) - sum_c (N_c / N) * I(c)

where I is Gini 1 - sum_k p_k^2, entropy -sum_k p_k log2 p_k, or the
regression impurity (1/N) sum_i (y_i - mean(y))^2.

scikit-learn is NOT used here; it is only the baseline we compare against.
"""

from __future__ import annotations

import numpy as np

CLASSIFICATION_CRITERIA = ("gini", "entropy")
REGRESSION_CRITERIA = ("mse",)


class Node:
    """One node of the tree. Internal nodes hold a split; every node holds a value."""
    def __init__(self, n_samples: int, impurity: float, value):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: x[feature] <= threshold goes left (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = value       # class probabilities (classification) or mean (regression)
        self.n_samples = n_samples
        self.impurity = impurity

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None
        self.classes_ = None      # original class labels (classification only)

    def _check_params(self) -> None:
        if self.task == "classification":
            allowed = CLASSIFICATION_CRITERIA
        elif self.task == "regression":
            allowed = REGRESSION_CRITERIA
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got {self.task!r}")
        if self.criterion not in allowed:
            raise ValueError(f"criterion for {self.task} must be one of {allowed}, "
                             f"got {self.criterion!r}")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be None or >= 0")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be >= 0")

    # --- impurity -----------------------------------------------------------
    def _impurity_from_counts(self, counts):
        """
        Gini or entropy from class counts. `counts` has shape (..., n_classes),
        so the same code scores one node or a whole grid of candidate splits.
        """
        n = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(n, 1)
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=-1)
        log_p = np.log2(np.where(p > 0, p, 1.0))   # 0 * log 0 is taken as 0
        return -np.sum(p * log_p, axis=-1)

    def _impurity(self, y) -> float:
        """Gini / entropy of encoded labels (classification) or MSE around the mean (regression)."""
        if self.task == "classification":
            counts = np.bincount(y, minlength=len(self.classes_))
            return float(self._impurity_from_counts(counts))
        return float(np.mean((y - y.mean()) ** 2))

    def _leaf_value(self, y):
        """Class frequencies (-> majority vote and predict_proba) or the mean target."""
        if self.task == "classification":
            return np.bincount(y, minlength=len(self.classes_)) / len(y)
        return float(y.mean())

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        raise NotImplementedError("TODO: best split search")

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")
