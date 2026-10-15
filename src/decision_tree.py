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
EPSILON = np.finfo(float).eps   # tolerance for "pure node" and decrease comparisons


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
        self.n_features_in_ = None  # number of columns seen by fit
        self._n_train = None      # rows seen by fit (weights min_impurity_decrease)
        self._importances = None  # unnormalised total impurity decrease per feature
        self.feature_importances_ = None

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
        for name in ("max_depth", "min_samples_split", "min_samples_leaf"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer, got {value!r}")
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
        return 0.0 - np.sum(p * log_p, axis=-1)   # 0.0 - x avoids printing -0.0 for pure nodes

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

    @staticmethod
    def _is_pure(y, impurity: float) -> bool:
        """
        A node is pure when its impurity is ~0 or all targets are identical.
        The second test matters for regression with large targets: the mean of
        identical values like 1.23e8 can be off by one ulp, which leaves an MSE
        of ~1e-14 that a fixed EPSILON does not catch.
        """
        return impurity <= EPSILON or bool(np.all(y == y[0]))

    def _best_split(self, X, y):
        """
        Search every feature and every candidate threshold of this node at once.

        Each feature is sorted once; a threshold between sorted positions i and
        i+1 sends the first i+1 rows left. Cumulative sums over the sorted rows
        give the class counts (classification) or sum / sum of squares
        (regression) of every left child in one pass, and the right child is
        "node total minus left". So one node costs O(n log n * d), not O(n^2 * d).

        Candidate thresholds are midpoints between consecutive DISTINCT values
        (as in sklearn); positions that leave fewer than min_samples_leaf rows
        on either side are invalid.

        Returns (feature, threshold, gain) with gain = I(parent) - weighted
        child impurity, or None if the node has no valid split. Ties go to the
        lowest feature index, then the lowest threshold.
        """
        n, _ = X.shape
        leaf = self.min_samples_leaf
        if n < 2 * leaf:
            return None

        order = np.argsort(X, axis=0, kind="stable")       # (n, d)
        X_sorted = np.take_along_axis(X, order, axis=0)
        n_left = np.arange(1, n, dtype=float)[:, None]      # row i: rows 0..i go left
        n_right = n - n_left

        if self.task == "classification":
            one_hot = np.eye(len(self.classes_))[y]           # (n, K)
            left_counts = np.cumsum(one_hot[order], axis=0)[:-1]   # (n-1, d, K)
            right_counts = one_hot.sum(axis=0) - left_counts
            child = (n_left * self._impurity_from_counts(left_counts)
                     + n_right * self._impurity_from_counts(right_counts)) / n
        else:
            y_c = y - y.mean()                  # centring keeps the sums numerically stable
            y_sorted = y_c[order]
            sum_left = np.cumsum(y_sorted, axis=0)[:-1]
            sq_left = np.cumsum(y_sorted ** 2, axis=0)[:-1]
            sum_right = y_c.sum() - sum_left
            sq_right = np.sum(y_c ** 2) - sq_left
            sse_left = sq_left - sum_left ** 2 / n_left       # sum of squared deviations
            sse_right = sq_right - sum_right ** 2 / n_right
            child = (sse_left + sse_right) / n

        valid = X_sorted[1:] > X_sorted[:-1]                # threshold must separate two values
        valid &= (n_left >= leaf) & (n_right >= leaf)
        if not valid.any():
            return None

        gain = np.where(valid, self._impurity(y) - child, -np.inf)
        feature, pos = np.unravel_index(np.argmax(gain.T), gain.T.shape)

        low, high = X_sorted[pos, feature], X_sorted[pos + 1, feature]
        threshold = low / 2.0 + high / 2.0      # (low + high) / 2 can overflow to inf
        if threshold == high or not np.isfinite(threshold):   # keep `high` on the right side
            threshold = low
        return int(feature), float(threshold), float(gain[pos, feature])

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """
        Recursively build the tree. A node becomes a leaf when
          - it reached max_depth,
          - it has fewer than min_samples_split rows,
          - it is pure (impurity 0),
          - no threshold leaves min_samples_leaf rows on both sides, or
          - the best split's weighted decrease (N_t / N) * gain is below
            min_impurity_decrease (sklearn's definition, N = training rows).
        """
        n = len(y)
        impurity = self._impurity(y)
        node = Node(n_samples=n, impurity=impurity, value=self._leaf_value(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n < self.min_samples_split or self._is_pure(y, impurity):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, gain = split
        if (n / self._n_train) * gain + EPSILON < self.min_impurity_decrease:
            return node

        go_left = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        self._importances[feature] += n * gain    # = N_t*I_t - N_left*I_left - N_right*I_right
        node.left = self._grow(X[go_left], y[go_left], depth + 1)
        node.right = self._grow(X[~go_left], y[~go_left], depth + 1)
        return node

    def fit(self, X, y):
        """Build the tree from X (n_samples, n_features) and y. Returns self."""
        self._check_params()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:     # accept a column vector like sklearn
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError(f"expected X of shape (n, d) and y of length n, "
                             f"got X {X.shape} and y {y.shape}")
        if np.isnan(X).any():
            raise ValueError("X contains NaN: impute first (data_prep.impute_median)")
        if np.isinf(X).any():
            raise ValueError("X contains inf")

        if self.task == "classification":
            if y.dtype.kind == "f" and np.isnan(y).any():
                raise ValueError("y contains NaN labels")
            self.classes_, y = np.unique(y, return_inverse=True)   # labels -> 0..K-1
        else:
            y = y.astype(float)
            if not np.isfinite(y).all():
                raise ValueError("y contains NaN or inf")
        self.n_features_in_ = X.shape[1]
        self._n_train = len(y)
        self._importances = np.zeros(X.shape[1])
        self.root = self._grow(X, y, depth=0)

        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        return self

    def _check_fitted(self) -> None:
        if self.root is None:
            raise RuntimeError("call fit first")

    def get_depth(self) -> int:
        """Length of the longest root-to-leaf path (a single leaf has depth 0)."""
        self._check_fitted()

        def depth(node: Node) -> int:
            if node.is_leaf():
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        return depth(self.root)

    def get_n_leaves(self) -> int:
        """Number of leaves."""
        self._check_fitted()

        def count(node: Node) -> int:
            if node.is_leaf():
                return 1
            return count(node.left) + count(node.right)
        return count(self.root)

    def _leaf_values(self, X):
        """
        Route all rows down the tree together and return each row's leaf value.
        Iterative (explicit stack) and vectorised per node: every node splits
        its block of row indices with one comparison, so there is no Python
        loop over rows.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected X of shape (n, {self.n_features_in_}) as in fit, "
                             f"got {X.shape}")
        if np.isnan(X).any():   # NaN <= t is False, so NaN rows would silently go right
            raise ValueError("X contains NaN: impute first (data_prep.impute_median)")
        if self.task == "classification":
            out = np.empty((len(X), len(self.classes_)))
        else:
            out = np.empty(len(X))

        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if len(rows) == 0:
                continue
            if node.is_leaf():
                out[rows] = node.value
                continue
            go_left = X[rows, node.feature] <= node.threshold
            stack.append((node.left, rows[go_left]))
            stack.append((node.right, rows[~go_left]))
        return out

    def predict(self, X):
        """Majority class of the leaf (classification) or the leaf mean (regression)."""
        values = self._leaf_values(X)
        if self.task == "classification":
            return self.classes_[np.argmax(values, axis=1)]
        return values

    def predict_proba(self, X):
        """Classification only: class frequencies of the leaf, columns ordered as classes_."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        return self._leaf_values(X)
