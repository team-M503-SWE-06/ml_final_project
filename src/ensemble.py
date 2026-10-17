"""
ensemble.py: Random Forest and gradient boosting built from OUR DecisionTree (Problem 9).

RandomForest (both tasks): many decorrelated trees, averaged.
  - each tree is fit on a bootstrap sample (n rows drawn with replacement),
  - each split looks at a random subset of max_features features,
  - classification averages the trees' class probabilities, regression
    averages their predictions.
  Averaging keeps the low bias of deep trees and cuts their variance. The
  rows a tree never saw (~37% per tree) give a free out-of-bag (OOB)
  estimate of the error.

GradientBoostingRegressor (Task A): shallow trees fit one after another to
the residuals of the current model (gradient boosting with squared loss):
    F_0 = mean(y),   F_m = F_{m-1} + learning_rate * tree_m(x),
    tree_m fit on r = y - F_{m-1}.
  Regularisation: learning_rate (shrinkage), shallow trees, subsample
  (stochastic boosting), and L2 / L1 on each tree's leaf values (leaf_l2,
  leaf_l1, the XGBoost leaf penalties, see DecisionTree._leaf_value).

scikit-learn is not used here; its RandomForest / GradientBoosting are the
baselines in src/experiments/ensembles.py.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np

from .decision_tree import DecisionTree


_DATA = {}


def _share_data(X, y):
    """Pool initializer: each worker receives the training data once."""
    _DATA["X"], _DATA["y"] = X, y


def _fit_tree(job):
    """Fit one forest tree on its bootstrap rows; top-level so worker processes can run it."""
    params, seed, rows = job
    return DecisionTree(**params, random_state=seed).fit(_DATA["X"][rows], _DATA["y"][rows])


class RandomForest:
    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 100,
        criterion: str | None = None,       # default: "gini" / "mse"
        max_features="sqrt",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        leaf_l2: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,           # processes for fitting trees; -1 = all cores
        random_state: int = 42,
    ):
        self.n_jobs = n_jobs
        self.task = task
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.leaf_l2 = leaf_l2
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees_ = []
        self.classes_ = None

    def fit(self, X, y):
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score needs bootstrap=True")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(y)
        if self.task == "classification":
            self.classes_, y_enc = np.unique(y, return_inverse=True)
        else:
            y_enc = y.astype(float)
        criterion = self.criterion or ("gini" if self.task == "classification" else "mse")
        rng = np.random.default_rng(self.random_state)
        params = dict(task=self.task, criterion=criterion, max_depth=self.max_depth,
                      min_samples_split=self.min_samples_split,
                      min_samples_leaf=self.min_samples_leaf,
                      min_impurity_decrease=self.min_impurity_decrease,
                      max_features=self.max_features, leaf_l2=self.leaf_l2)

        # draw every bootstrap sample and tree seed up front, so the forest is
        # identical whether the trees are fit in one process or in parallel
        samples = [rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
                   for _ in range(self.n_estimators)]
        seeds = [int(s) for s in rng.integers(2 ** 31 - 1, size=self.n_estimators)]
        jobs = [(params, seed, rows) for rows, seed in zip(samples, seeds)]
        n_jobs = os.cpu_count() if self.n_jobs == -1 else (self.n_jobs or 1)
        if n_jobs > 1:
            with Pool(min(n_jobs, self.n_estimators), initializer=_share_data,
                      initargs=(X, y_enc)) as pool:
                self.trees_ = pool.map(_fit_tree, jobs)
        else:
            _share_data(X, y_enc)
            self.trees_ = [_fit_tree(job) for job in jobs]
            _DATA.clear()

        oob_sum = np.zeros((n, len(self.classes_))) if self.task == "classification" else np.zeros(n)
        oob_count = np.zeros(n)
        if self.oob_score:
            for tree, rows in zip(self.trees_, samples):
                out = np.setdiff1d(np.arange(n), rows)
                oob_sum[out] += self._tree_output(tree, X[out])
                oob_count[out] += 1

        importances = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        if self.oob_score:
            seen = oob_count > 0
            if self.task == "classification":
                self.oob_decision_function_ = oob_sum[seen] / oob_count[seen, None]
                self.oob_score_ = float(np.mean(self.oob_decision_function_.argmax(axis=1)
                                                == y_enc[seen]))           # accuracy
            else:
                self.oob_prediction_ = oob_sum[seen] / oob_count[seen]
                resid = y_enc[seen] - self.oob_prediction_
                self.oob_score_ = float(1 - np.sum(resid ** 2)
                                        / np.sum((y_enc[seen] - y_enc[seen].mean()) ** 2))  # R2
            self.oob_rows_ = np.flatnonzero(seen)
        return self

    def _tree_output(self, tree: DecisionTree, X):
        """Class probabilities over ALL forest classes (a bootstrap may miss one), or values."""
        if self.task != "classification":
            return tree.predict(X)
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, tree.classes_] = tree.predict_proba(X)     # tree.classes_ are forest codes
        return proba

    def staged_output(self, X):
        """Yield the averaged output after 1, 2, ..., n_estimators trees."""
        if not self.trees_:
            raise RuntimeError("call fit first")
        X = np.asarray(X, dtype=float)
        total = 0.0
        for k, tree in enumerate(self.trees_, 1):
            total = total + self._tree_output(tree, X)
            yield total / k

    def _output(self, X):
        for out in self.staged_output(X):
            pass
        return out

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        return self._output(X)

    def predict(self, X):
        out = self._output(X)
        if self.task == "classification":
            return self.classes_[np.argmax(out, axis=1)]
        return out

    def staged_predict(self, X):
        """Predictions with the first k trees, k = 1..n_estimators (error vs number of trees)."""
        for out in self.staged_output(X):
            yield self.classes_[np.argmax(out, axis=1)] if self.task == "classification" else out


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features=None,
        leaf_l2: float = 0.0,
        leaf_l1: float = 0.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.leaf_l2 = leaf_l2
        self.leaf_l1 = leaf_l1
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        if self.n_estimators < 1 or not 0 < self.learning_rate <= 1 or not 0 < self.subsample <= 1:
            raise ValueError("need n_estimators >= 1, 0 < learning_rate <= 1, 0 < subsample <= 1")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(y.mean())
        f = np.full(n, self.init_)
        self.trees_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            residual = y - f                    # negative gradient of (y - f)^2 / 2
            rows = (rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)
                    if self.subsample < 1 else np.arange(n))
            tree = DecisionTree(
                task="regression", criterion="mse", max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                leaf_l2=self.leaf_l2, leaf_l1=self.leaf_l1, random_state=int(rng.integers(2 ** 31 - 1)),
            ).fit(X[rows], residual[rows])
            f += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(float(np.mean((y - f) ** 2)))
        importances = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def staged_predict(self, X):
        """Predictions after 1, 2, ..., n_estimators boosting steps."""
        if not self.trees_:
            raise RuntimeError("call fit first")
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.init_)
        for tree in self.trees_:
            f = f + self.learning_rate * tree.predict(X)
            yield f

    def predict(self, X):
        for f in self.staged_predict(X):
            pass
        return f
