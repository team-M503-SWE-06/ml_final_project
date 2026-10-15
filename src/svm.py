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
