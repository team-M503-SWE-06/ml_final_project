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
        # computed on the X that was trained on (not via decision_function, so a
        # subclass that maps X -> phi(X) in decision_function is not mapped twice)
        self.n_support_ = int(np.sum(y_pm1 * (X @ w + b) <= 1.0))
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


class RFFPegasosSVM(PegasosSVM):
    """
    Non-linear SVM (Problem 3d, option ii): random Fourier features + linear Pegasos.

    The RBF kernel k(x, x') = exp(-gamma * ||x - x'||^2) is approximated by an
    explicit map (Rahimi & Recht, 2007)

        phi(x) = sqrt(2 / D) * cos(W^T x + c),   W_jk ~ N(0, 2*gamma),  c_k ~ U(0, 2*pi)

    so that phi(x) . phi(x') ~= k(x, x'), and the unchanged linear PegasosSVM is
    trained on phi(X).

    Why RFF and not kernelised Pegasos: kernel Pegasos keeps one alpha_i per
    training row and needs K(x_i, x_j) against every row at each step, i.e.
    O(n^2) kernel evaluations per epoch (~2e9 for our ~45k training rows) and
    an O(n) cost per prediction. RFF costs O(n * D) per epoch, reuses the
    tested linear solver as-is, and predicts in O(D).

    W and c are drawn once in fit (seeded by random_state) and reused for
    every later call. norm_w_ / margin_width_ / support vectors are measured
    in the D-dimensional feature space.
    """

    def __init__(
        self,
        n_components: int = 500,
        gamma: float = 0.01,
        lambda_: float = 1e-4,
        n_epochs: int = 10,
        lr_schedule: str = "pegasos",
        eta0: float = 0.01,
        project: bool = True,
        random_state: int = 42,
    ):
        super().__init__(lambda_=lambda_, n_epochs=n_epochs, lr_schedule=lr_schedule,
                         eta0=eta0, project=project, random_state=random_state)
        self.n_components = n_components
        self.gamma = gamma
        self.W_ = None          # (d, D) random frequencies
        self.offset_ = None     # (D,)   random phases

    def _features(self, X) -> np.ndarray:
        """phi(X) = sqrt(2/D) * cos(X W + c)."""
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W_ + self.offset_)

    def _check_X(self, X) -> np.ndarray:
        """Validate raw X against the fitted input width and return phi(X)."""
        if self.W_ is None:
            raise RuntimeError("RFFPegasosSVM is not fitted yet; call fit(X, y) first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.W_.shape[0]:
            raise ValueError(f"X must have shape (n, {self.W_.shape[0]}), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf values")
        return self._features(X)

    def fit(self, X, y):
        """Draw W, c, map X -> phi(X), then run the linear Pegasos fit on phi(X)."""
        if not isinstance(self.n_components, (int, np.integer)) or isinstance(self.n_components, bool) \
                or self.n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if not self.gamma > 0:
            raise ValueError("gamma must be > 0")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a 2-D finite array without NaNs/infs (impute + standardize first)")

        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(X.shape[1], self.n_components))
        self.offset_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return super().fit(self._features(X), y)
