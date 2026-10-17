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

import heapq
import inspect
import itertools

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
        max_features=None,              # None | int | float in (0, 1] | "sqrt" | "log2"
        max_leaf_nodes: int | None = None,   # grow best-first up to this many leaves
        leaf_l2: float = 0.0,           # L2 shrinkage of leaf values (see _leaf_value)
        leaf_l1: float = 0.0,           # L1 soft-threshold of leaf values (regression)
        ccp_alpha: float = 0.0,         # cost-complexity pruning strength (0 = no pruning)
        random_state: int = 42,
    ):
        self.ccp_alpha = ccp_alpha
        self.max_leaf_nodes = max_leaf_nodes
        self.leaf_l2 = leaf_l2
        self.leaf_l1 = leaf_l1
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
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
        mf = self.max_features
        if not (mf is None or mf in ("sqrt", "log2")
                or (isinstance(mf, (int, np.integer)) and not isinstance(mf, bool) and mf >= 1)
                or (isinstance(mf, float) and 0.0 < mf <= 1.0)):
            raise ValueError("max_features must be None, 'sqrt', 'log2', an int >= 1 "
                             f"or a float in (0, 1], got {mf!r}")
        mln = self.max_leaf_nodes
        if mln is not None and (not isinstance(mln, (int, np.integer)) or mln < 2):
            raise ValueError(f"max_leaf_nodes must be None or an integer >= 2, got {mln!r}")
        if self.leaf_l2 < 0 or self.leaf_l1 < 0:
            raise ValueError("leaf_l2 and leaf_l1 must be >= 0")
        if self.leaf_l1 > 0 and self.task == "classification":
            raise ValueError("leaf_l1 applies to regression leaves only; use leaf_l2 to "
                             "smooth class probabilities")
        if self.ccp_alpha < 0:
            raise ValueError("ccp_alpha must be >= 0")

    def _features_per_split(self, n_features: int) -> int:
        """How many randomly chosen features each split looks at (all of them by default)."""
        mf = self.max_features
        if mf is None:
            k = n_features
        elif mf == "sqrt":
            k = int(np.sqrt(n_features))
        elif mf == "log2":
            k = int(np.log2(n_features))
        elif isinstance(mf, float):
            k = int(mf * n_features)
        else:
            k = mf
        return min(max(k, 1), n_features)

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
        """
        Class frequencies (-> majority vote and predict_proba) or the mean target.

        Optional L1/L2 regularisation of the value (the split search is not
        affected). It pulls small leaves towards the training-set answer, so a
        leaf with 3 rows cannot make an extreme prediction on its own:
          - regression, with G = sum(y - y_train_mean) over the leaf's rows:
                value = y_train_mean + sign(G) * max(|G| - leaf_l1, 0) / (n + leaf_l2)
            (the XGBoost leaf formula for squared loss: L2 shrinks the leaf's
            offset, L1 sets small offsets to exactly zero);
          - classification: (counts + leaf_l2 * prior) / (n + leaf_l2), i.e.
            probabilities smoothed towards the training class frequencies.
        With leaf_l2 = leaf_l1 = 0 this is the plain frequency / mean.
        """
        n = len(y)
        if self.task == "classification":
            counts = np.bincount(y, minlength=len(self.classes_))
            if self.leaf_l2 == 0:
                return counts / n
            return (counts + self.leaf_l2 * self._prior) / (n + self.leaf_l2)
        if self.leaf_l2 == 0 and self.leaf_l1 == 0:
            return float(y.mean())
        g = float(np.sum(y - self._y_train_mean))
        g = np.sign(g) * max(abs(g) - self.leaf_l1, 0.0)
        return float(self._y_train_mean + g / (n + self.leaf_l2))

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
        Best split of this node over all features, or over a random subset of
        max_features of them (the Random Forest trick that decorrelates trees).
        Like sklearn, if none of the drawn features has a valid split, the
        remaining features are searched too, so the subset never forces an
        early leaf. Returns (feature, threshold, gain) or None.
        """
        n_features = X.shape[1]
        if self._k_features >= n_features:
            return self._search_split(X, y)
        order = self._rng.permutation(n_features)
        for cols in (np.sort(order[:self._k_features]), np.sort(order[self._k_features:])):
            split = self._search_split(X[:, cols], y)
            if split is not None:
                feature, threshold, gain = split
                return int(cols[feature]), threshold, gain
        return None

    def _search_split(self, X, y):
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
    def _new_node(self, y) -> Node:
        return Node(n_samples=len(y), impurity=self._impurity(y), value=self._leaf_value(y))

    def _choose_split(self, X, y, node: Node, depth: int):
        """
        Decide whether `node` (holding rows X, y) is split. It stays a leaf when
          - it reached max_depth,
          - it has fewer than min_samples_split rows,
          - it is pure,
          - no threshold leaves min_samples_leaf rows on both sides, or
          - the best split's weighted decrease (N_t / N) * gain is below
            min_impurity_decrease (sklearn's definition, N = training rows).
        Returns (feature, threshold, gain) or None.
        """
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        if node.n_samples < self.min_samples_split or self._is_pure(y, node.impurity):
            return None
        split = self._best_split(X, y)
        if split is None:
            return None
        if (node.n_samples / self._n_train) * split[2] + EPSILON < self.min_impurity_decrease:
            return None
        return split

    def _grow(self, X, y) -> Node:
        """
        Build the tree top-down by recursive partitioning: split a node, then
        split each child the same way, until every branch hits a stopping rule.

        The recursion is written with an explicit stack of (node, row indices,
        depth) instead of Python function calls, so a very deep tree
        (max_depth=None on chain-like data) cannot hit Python's recursion limit
        of ~1000, and children get row indices instead of copies of X.

        With max_leaf_nodes set the tree grows best-first instead (as sklearn
        does): the frontier is a priority queue and the leaf whose best split
        removes the most impurity (N_t * gain) is split next, wherever it is,
        until the tree has max_leaf_nodes leaves.
        """
        root = self._new_node(y)
        all_rows = np.arange(len(y))
        if self.max_leaf_nodes is not None:
            return self._grow_best_first(X, y, root, all_rows)

        stack = [(root, all_rows, 0)]
        while stack:
            node, rows, depth = stack.pop()
            split = self._choose_split(X[rows], y[rows], node, depth)
            if split is None:
                continue
            (left, left_rows), (right, right_rows) = self._apply_split(X, y, node, rows, split)
            stack.append((right, right_rows, depth + 1))
            stack.append((left, left_rows, depth + 1))
        return root

    def _grow_best_first(self, X, y, root: Node, all_rows) -> Node:
        heap, tie = [], itertools.count()     # tie: equal priorities pop in creation order

        def push(node, rows, depth):
            split = self._choose_split(X[rows], y[rows], node, depth)
            if split is not None:
                heapq.heappush(heap, (-node.n_samples * split[2], next(tie), node, rows, depth, split))

        push(root, all_rows, 0)
        n_leaves = 1
        while heap and n_leaves < self.max_leaf_nodes:
            _, _, node, rows, depth, split = heapq.heappop(heap)
            for child, child_rows in self._apply_split(X, y, node, rows, split):
                push(child, child_rows, depth + 1)
            n_leaves += 1
        return root

    def _apply_split(self, X, y, node: Node, rows, split):
        """Turn `node` into an internal node; return (child, child_rows) for both sides."""
        feature, threshold, gain = split
        node.feature, node.threshold = feature, threshold
        self._importances[feature] += node.n_samples * gain   # = N_t*I_t - N_L*I_L - N_R*I_R
        go_left = X[rows, feature] <= threshold
        left_rows, right_rows = rows[go_left], rows[~go_left]
        node.left, node.right = self._new_node(y[left_rows]), self._new_node(y[right_rows])
        return (node.left, left_rows), (node.right, right_rows)

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
        if self.task == "classification":
            self._prior = np.bincount(y, minlength=len(self.classes_)) / len(y)
        else:
            self._y_train_mean = float(y.mean())
        self.n_features_in_ = X.shape[1]
        self._k_features = self._features_per_split(X.shape[1])
        self._rng = np.random.default_rng(self.random_state)   # only used with max_features
        self._n_train = len(y)
        self._importances = np.zeros(X.shape[1])
        self.root = self._grow(X, y)
        if self.ccp_alpha > 0:
            self._prune(self.ccp_alpha)
            self._recompute_importances()

        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        return self

    # --- cost-complexity pruning (Breiman et al., CART) -----------------------
    def _prune(self, alpha: float, path: list | None = None) -> None:
        """
        Minimal cost-complexity pruning, the post-pruning sklearn calls ccp_alpha.

        The grown tree T is scored by R_alpha(T) = R(T) + alpha * |leaves(T)|,
        where R(T) = sum over leaves of (N_t / N) * impurity_t. Collapsing the
        subtree under node t into one leaf raises R by R(t) - R(T_t) and
        removes |leaves(T_t)| - 1 leaves, so it pays off once alpha exceeds
        the node's effective alpha

            g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1).

        We repeatedly collapse the "weakest link" (smallest g) while g <= alpha.
        Collapsing t only changes g for t's ancestors, so a heap with version
        numbers keeps this O(n log n) instead of rescanning the tree each time.
        With `path` given, (g, R(T)) is appended after every collapse.
        """
        def risk(node):
            return node.n_samples / self._n_train * node.impurity

        parent, order, stack = {}, [], [self.root]
        while stack:                                   # pre-order walk
            node = stack.pop()
            order.append(node)
            if not node.is_leaf():
                parent[id(node.left)] = parent[id(node.right)] = node
                stack += [node.right, node.left]
        leaves, sub_risk = {}, {}                      # |leaves(T_t)|, R(T_t)
        for node in reversed(order):                   # children before parents
            if node.is_leaf():
                leaves[id(node)], sub_risk[id(node)] = 1, risk(node)
            else:
                leaves[id(node)] = leaves[id(node.left)] + leaves[id(node.right)]
                sub_risk[id(node)] = sub_risk[id(node.left)] + sub_risk[id(node.right)]

        def effective_alpha(node):
            return (risk(node) - sub_risk[id(node)]) / (leaves[id(node)] - 1)

        version, tie, heap = {}, itertools.count(), []

        def push(node):
            version[id(node)] = version.get(id(node), -1) + 1
            heapq.heappush(heap, (effective_alpha(node), next(tie), version[id(node)], node))

        for node in order:
            if not node.is_leaf():
                push(node)
        removed = set()                                # nodes inside collapsed subtrees
        while heap:
            g, _, v, node = heapq.heappop(heap)
            if node.is_leaf() or v != version[id(node)] or id(node) in removed:
                continue
            if g > alpha:
                break
            removed_leaves = leaves[id(node)] - 1
            added_risk = risk(node) - sub_risk[id(node)]
            below = [node.left, node.right]
            while below:
                child = below.pop()
                removed.add(id(child))
                if not child.is_leaf():
                    below += [child.left, child.right]
            node.left = node.right = node.feature = node.threshold = None
            leaves[id(node)], sub_risk[id(node)] = 1, risk(node)
            ancestor = parent.get(id(node))
            while ancestor is not None:
                leaves[id(ancestor)] -= removed_leaves
                sub_risk[id(ancestor)] += added_risk
                push(ancestor)
                ancestor = parent.get(id(ancestor))
            if path is not None:
                path.append((g, sub_risk[id(self.root)]))

    def prune(self, alpha: float):
        """
        Prune this fitted tree in place with cost-complexity strength `alpha`
        (same result as fitting with ccp_alpha=alpha). Pruning a deep copy of
        one grown tree at several alphas is much cheaper than refitting.
        """
        self._check_fitted()
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self._prune(alpha)
        self._recompute_importances()
        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        self.ccp_alpha = alpha
        return self

    def _recompute_importances(self) -> None:
        """Impurity-decrease importances of the internal nodes that survived pruning."""
        self._importances = np.zeros(self.n_features_in_)
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                continue
            self._importances[node.feature] += (
                node.n_samples * node.impurity - node.left.n_samples * node.left.impurity
                - node.right.n_samples * node.right.impurity)
            stack += [node.left, node.right]

    def cost_complexity_pruning_path(self, X, y) -> dict:
        """
        The weakest-link pruning sequence of the fully grown tree (same
        parameters, ccp_alpha ignored), like sklearn's method of the same name:
        {"ccp_alphas": effective alphas in increasing order, starting at 0,
         "impurities": total leaf impurity R(T) of the tree after each step}.
        Pick ccp_alpha from these values with validation data.
        """
        params = {name: getattr(self, name) for name in inspect.signature(DecisionTree).parameters}
        params["ccp_alpha"] = 0.0
        full = DecisionTree(**params).fit(X, y)
        path = [(0.0, sum(full._leaf_risks()))]
        full._prune(np.inf, path)
        alphas, impurities = zip(*path)
        return {"ccp_alphas": np.array(alphas), "impurities": np.array(impurities)}

    def _leaf_risks(self) -> list[float]:
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                out.append(node.n_samples / self._n_train * node.impurity)
            else:
                stack += [node.left, node.right]
        return out

    def _check_fitted(self) -> None:
        if self.root is None:
            raise RuntimeError("call fit first")

    def _leaf_depths(self) -> list[int]:
        """Depth of every leaf, walking the tree with an explicit stack."""
        self._check_fitted()
        depths, stack = [], [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if node.is_leaf():
                depths.append(depth)
            else:
                stack.append((node.left, depth + 1))
                stack.append((node.right, depth + 1))
        return depths

    def get_depth(self) -> int:
        """Length of the longest root-to-leaf path (a single leaf has depth 0)."""
        return max(self._leaf_depths())

    def get_n_leaves(self) -> int:
        """Number of leaves."""
        return len(self._leaf_depths())

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
