import numpy as np
from sklearn import metrics as skm
from src import evaluate as ev

rng = np.random.default_rng(0)
yt = rng.normal(size=200)
yp = yt + rng.normal(scale=.5, size=200)
yb = rng.integers(0, 2, 200)
sc = rng.random(200) + yb * .3
yhat = (sc > .5).astype(int)


def test_regression():
    assert np.isclose(ev.rmse(yt, yp), skm.mean_squared_error(yt, yp) ** .5)
    assert np.isclose(ev.mae(yt, yp), skm.mean_absolute_error(yt, yp))
    assert np.isclose(ev.r2(yt, yp), skm.r2_score(yt, yp))


def test_classification():
    assert (ev.confusion_matrix(yb, yhat) == skm.confusion_matrix(yb, yhat)).all()
    p, r, f = ev.precision_recall_f1(yb, yhat)
    assert np.isclose(p, skm.precision_score(yb, yhat))
    assert np.isclose(r, skm.recall_score(yb, yhat))
    assert np.isclose(f, skm.f1_score(yb, yhat))
    assert np.isclose(ev.roc_auc(yb, sc), skm.roc_auc_score(yb, sc))