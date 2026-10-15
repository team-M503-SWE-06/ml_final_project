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

    def fit(self, X, y):
        raise NotImplementedError("TODO: Pegasos training loop")

    def decision_function(self, X):
        raise NotImplementedError("TODO: decision_function")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")
