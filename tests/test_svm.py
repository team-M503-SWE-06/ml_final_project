"""Tests for the from-scratch Pegasos SVM (synthetic data only, no dataset needed)."""

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

from src.svm import PegasosSVM


def _blobs(n=300, d=4, sep=1.5, seed=42):
    """Two Gaussian blobs, labels {0,1}, standardized."""
    rng = np.random.default_rng(seed)
    h = n // 2
    X = np.vstack([rng.normal(-sep, 1.0, (h, d)), rng.normal(sep, 1.0, (n - h, d))])
    y = np.r_[np.zeros(h, int), np.ones(n - h, int)]
    p = rng.permutation(n)
    X, y = X[p], y[p]
    return (X - X.mean(0)) / X.std(0), y


def test_to_pm1_mapping_and_classes():
    y_pm1, classes = PegasosSVM._to_pm1([0, 1, 1, 0])
    assert np.array_equal(y_pm1, [-1, 1, 1, -1])
    assert np.array_equal(classes, [0, 1])
    y_pm1, classes = PegasosSVM._to_pm1(np.array(["std", "prem", "prem"]))
    assert np.array_equal(classes, ["prem", "std"])
    assert np.array_equal(y_pm1, [1, -1, -1])
    with pytest.raises(ValueError):
        PegasosSVM._to_pm1([1, 1, 1])


def test_fit_predict_keeps_original_labels():
    X, y = _blobs()
    m = PegasosSVM(lambda_=0.01, n_epochs=10, random_state=0)
    assert m.fit(X, y) is m
    assert m.w.shape == (4,) and isinstance(m.b, float)
    pred = m.predict(X)
    assert set(np.unique(pred)) <= {0, 1}
    assert np.mean(pred == y) >= 0.95

    y_pm = np.where(y == 1, 1, -1)
    pred_pm = PegasosSVM(lambda_=0.01, random_state=0).fit(X, y_pm).predict(X)
    assert set(np.unique(pred_pm)) <= {-1, 1}
    assert np.mean(pred_pm == y_pm) >= 0.95


def test_predict_matches_sign_of_decision_function():
    X, y = _blobs(seed=1)
    m = PegasosSVM(lambda_=0.01, random_state=1).fit(X, y)
    s = m.decision_function(X)
    assert np.array_equal(m.predict(X), np.where(s >= 0, 1, 0))


def test_deterministic_with_seed():
    X, y = _blobs(seed=2)
    a = PegasosSVM(lambda_=0.001, random_state=7).fit(X, y)
    b = PegasosSVM(lambda_=0.001, random_state=7).fit(X, y)
    assert np.array_equal(a.w, b.w) and a.b == b.b


def test_objective_decreases():
    X, y = _blobs(n=400, sep=0.7, seed=3)
    m = PegasosSVM(lambda_=0.01, n_epochs=20, random_state=3).fit(X, y)
    h = np.array(m.loss_history_)
    assert len(h) == 20
    assert h[-1] < h[0]
    assert h[-5:].std() < 0.05 * h[-1]


def test_margin_and_support_vectors():
    X, y = _blobs(n=400, sep=0.7, seed=4)
    m = PegasosSVM(lambda_=0.01, random_state=4).fit(X, y)
    assert np.isclose(m.norm_w_, np.linalg.norm(m.w))
    assert np.isclose(m.margin_width_, 2 / np.linalg.norm(m.w))
    mask = m.support_vector_mask(X, y)
    yf = np.where(y == 1, 1, -1) * m.decision_function(X)
    assert np.array_equal(mask, yf <= 1)
    assert m.n_support_ == mask.sum() and 0 < m.n_support_ < len(y)
    with pytest.raises(ValueError):
        m.support_vector_mask(X, np.full(len(y), 7))


def test_larger_lambda_gives_wider_margin():
    X, y = _blobs(n=400, sep=0.7, seed=5)
    fits = [PegasosSVM(lambda_=lam, random_state=5).fit(X, y) for lam in (1e-4, 1e-2, 1.0)]
    margins = [m.margin_width_ for m in fits]
    assert margins[0] < margins[1] < margins[2]
    n_sv = [m.n_support_ for m in fits]
    assert n_sv[0] <= n_sv[1] <= n_sv[2]       # wider margin -> more points inside it


@pytest.mark.parametrize("schedule", ["pegasos", "invsqrt", "constant"])
def test_lr_schedules_train(schedule):
    X, y = _blobs(seed=6)
    m = PegasosSVM(lambda_=0.01, lr_schedule=schedule, eta0=0.05, random_state=6).fit(X, y)
    assert np.mean(m.predict(X) == y) >= 0.95


def test_sanity_vs_sklearn_toy_split():
    """End-of-week-1 milestone: sanity test against sklearn on a toy split."""
    X, y = _blobs(n=600, d=5, sep=0.8, seed=123)
    Xtr, Xte, ytr, yte = X[:450], X[450:], y[:450], y[450:]
    lam = 0.001
    ours = PegasosSVM(lambda_=lam, n_epochs=20, random_state=42).fit(Xtr, ytr)
    sk = SGDClassifier(
        loss="hinge", alpha=lam, max_iter=1000, tol=1e-6, random_state=42
    ).fit(Xtr, ytr)
    acc_ours = np.mean(ours.predict(Xte) == yte)
    acc_sk = np.mean(sk.predict(Xte) == yte)
    assert abs(acc_ours - acc_sk) <= 0.05
    assert np.mean(ours.predict(Xte) == sk.predict(Xte)) >= 0.9
    cos = (ours.w @ sk.coef_.ravel()) / (np.linalg.norm(ours.w) * np.linalg.norm(sk.coef_))
    assert cos > 0.9


def test_input_validation():
    X, y = _blobs(n=50, seed=8)
    with pytest.raises(RuntimeError):
        PegasosSVM().predict(X)
    with pytest.raises(ValueError):
        PegasosSVM(lambda_=0).fit(X, y)
    with pytest.raises(ValueError):
        PegasosSVM(n_epochs=0).fit(X, y)
    with pytest.raises(ValueError):
        PegasosSVM(n_epochs=2.5).fit(X, y)
    with pytest.raises(ValueError):
        PegasosSVM(lr_schedule="adam").fit(X, y)
    Xn = X.copy(); Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        PegasosSVM().fit(Xn, y)
    Xi = X.copy(); Xi[0, 0] = np.inf
    with pytest.raises(ValueError):
        PegasosSVM().fit(Xi, y)
    m = PegasosSVM(lambda_=0.01).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :2])
    with pytest.raises(ValueError):
        m.predict(Xn)
