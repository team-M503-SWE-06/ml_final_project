"""
svm.py — linear soft-margin SVM trained with Pegasos, from scratch (NumPy only).

PegasosSVM minimises the regularised hinge-loss primal by stochastic
sub-gradient descent (Shalev-Shwartz et al., 2011), so no QP / SMO solver is
needed. Any two class labels are accepted: they are mapped to {-1, +1} for
training and mapped back in predict. The class docstring gives the exact
objective (the bias is regularised as a constant-1 feature) and fit() the
update rule; the objective is recorded after every epoch in loss_history_.

Inputs must be imputed and standardized with TRAIN statistics first
(data_prep.impute_median, data_prep.standardize): the hinge loss is
scale-sensitive. scikit-learn is used only for baselines elsewhere, never here.
"""

from __future__ import annotations

import numpy as np


LR_SCHEDULES = ("pegasos", "invsqrt", "constant")


class PegasosSVM:
    """
    Linear soft-margin SVM trained with Pegasos (stochastic sub-gradient).

    Bias handling: we use the trick from the statement's box — the bias is
    treated as the weight of a constant-1 feature. That means b is shrunk by
    the same (1 - eta_t*lambda) factor as w, i.e. the objective we actually
    minimise (and record in ``loss_history_``) is

        J(w, b) = (lambda/2) * (||w||^2 + b^2) + (1/n) * sum_i hinge_i

    With standardized features the extra b^2 term is negligible, and it keeps
    the very large early steps (eta_1 = 1/lambda) from blowing b up.

    Fitted attributes
    -----------------
    w, b              : weight vector and bias
    classes_          : original labels, classes_[1] is the "+1" class
    loss_history_     : objective J after every epoch (list of floats)
    norm_w_           : ||w||
    margin_width_     : 2 / ||w||   (geometric margin width)
    n_support_        : number of training points with y_i f(x_i) <= 1
    """

    def __init__(
        self,
        lambda_: float = 1e-4,
        n_epochs: int = 10,
        lr_schedule: str = "pegasos",
        eta0: float = 0.01,
        project: bool = True,
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.project = project
        self.random_state = random_state

        self.w = None
        self.b = 0.0
        self.classes_ = None
        self.loss_history_ = []

    @staticmethod
    def _to_pm1(y):
        """
        Map any 2-class labels to {-1, +1}.
        Returns (y_pm1, classes) where classes[0] -> -1 and classes[1] -> +1
        (classes are sorted, so {0,1} maps 0 -> -1, 1 -> +1).
        """
        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"PegasosSVM needs exactly 2 classes, got {classes.tolist()}")
        return np.where(y == classes[1], 1.0, -1.0), classes

    def _eta(self, t: int) -> float:
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr_schedule == "invsqrt":
            return self.eta0 / np.sqrt(t)
        return self.eta0

    def _objective(self, X, y_pm1) -> float:
        """J(w, b) as defined in the class docstring (bias regularised)."""
        hinge = np.maximum(0.0, 1.0 - y_pm1 * (X @ self.w + self.b))
        reg = 0.5 * self.lambda_ * (self.w @ self.w + self.b ** 2)
        return float(reg + hinge.mean())

    def _check_X(self, X) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("PegasosSVM is not fitted yet; call fit(X, y) first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError(f"X must have shape (n, {self.w.shape[0]}), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf values")
        return X

    def fit(self, X, y):
        """
        Run Pegasos for ``n_epochs`` shuffled passes (one example per step):

            eta_t = schedule(t)
            if y_i (w.x_i + b) < 1:  w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i
            else:                    w <- (1 - eta_t*lambda) w
            (same rule for b, see class docstring)
            optional: project (w, b) onto ||.|| <= 1/sqrt(lambda)

        The objective J is recorded after every epoch in ``loss_history_``.
        Return self.
        """
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        if not isinstance(self.n_epochs, (int, np.integer)) or isinstance(self.n_epochs, bool) or self.n_epochs < 1:
            raise ValueError("n_epochs must be a positive integer")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        if self.lr_schedule != "pegasos" and (self.eta0 <= 0 or self.eta0 * self.lambda_ >= 1):
            raise ValueError("need 0 < eta0 * lambda_ < 1, otherwise the shrink factor is <= 0")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a 2-D finite array without NaNs/infs (impute + standardize first)")
        y_pm1, self.classes_ = self._to_pm1(y)
        if len(y_pm1) != len(X):
            raise ValueError("X and y have different lengths")

        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        radius = 1.0 / np.sqrt(self.lambda_)
        rng = np.random.default_rng(self.random_state)
        self.loss_history_ = []

        t = 0
        for _ in range(self.n_epochs):
            for i in rng.permutation(n):
                t += 1
                eta = self._eta(t)
                shrink = 1.0 - eta * self.lambda_
                xi, yi = X[i], y_pm1[i]
                active = yi * (xi @ w + b) < 1.0
                w *= shrink
                b *= shrink
                if active:
                    w += eta * yi * xi
                    b += eta * yi
                if self.project:
                    norm = np.sqrt(w @ w + b * b)
                    if norm > radius:
                        w *= radius / norm
                        b *= radius / norm
            self.w, self.b = w, float(b)
            self.loss_history_.append(self._objective(X, y_pm1))

        self.w, self.b = w, float(b)
        self.n_support_ = int(self.support_vector_mask(X, y).sum())
        return self

    def decision_function(self, X) -> np.ndarray:
        """Return the signed score  w . x + b  for each row."""
        X = self._check_X(X)
        return X @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        """Map sign(w.x + b) back to the original labels (score 0 -> positive class)."""
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    # --- margin (Problem 3c) ------------------------------------------------
    @property
    def norm_w_(self) -> float:
        """||w|| (the bias is not part of the margin)."""
        if self.w is None:
            raise RuntimeError("PegasosSVM is not fitted yet")
        return float(np.linalg.norm(self.w))

    @property
    def margin_width_(self) -> float:
        """Geometric margin width 2 / ||w|| (inf if w == 0)."""
        nw = self.norm_w_
        return float("inf") if nw == 0 else 2.0 / nw

    def support_vector_mask(self, X, y) -> np.ndarray:
        """Boolean mask of support vectors: points with y_i (w.x_i + b) <= 1."""
        scores = self.decision_function(X)
        y = np.asarray(y).ravel()
        if len(y) != len(scores):
            raise ValueError("X and y have different lengths")
        unknown = ~np.isin(y, self.classes_)
        if unknown.any():
            raise ValueError(f"y has labels not seen in fit: {np.unique(y[unknown]).tolist()}")
        y_pm1 = np.where(y == self.classes_[1], 1.0, -1.0)
        return y_pm1 * scores <= 1.0
