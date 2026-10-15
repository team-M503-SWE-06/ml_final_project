"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
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

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_pm1, self.classes_ = self._to_pm1(y)
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
        return self

    def decision_function(self, X) -> np.ndarray:
        """Return the signed score  w . x + b  for each row."""
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        """Map sign(w.x + b) back to the original labels (score 0 -> positive class)."""
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
