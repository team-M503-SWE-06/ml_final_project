import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor as SkGB
from src import evaluate as ev
from src.decision_tree import DecisionTree
from src.ensemble import GradientBoostingRegressor, RandomForest


def _cls_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.7, size=n) > 0).astype(int)
    return X, y


def _reg_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.3, size=n)
    return X, y


def test_forest_beats_a_single_tree_on_held_out_data():
    X, y = _cls_data()
    forest = RandomForest(n_estimators=30, random_state=0).fit(X[:400], y[:400])
    tree = DecisionTree(criterion="gini").fit(X[:400], y[:400])
    assert np.mean(forest.predict(X[400:]) == y[400:]) > np.mean(tree.predict(X[400:]) == y[400:])
    Xr, yr = _reg_data()
    forest_r = RandomForest(task="regression", n_estimators=30, max_features=0.5,
                            random_state=0).fit(Xr[:400], yr[:400])
    tree_r = DecisionTree(task="regression", criterion="mse").fit(Xr[:400], yr[:400])
    assert ev.rmse(yr[400:], forest_r.predict(Xr[400:])) < ev.rmse(yr[400:], tree_r.predict(Xr[400:]))


def test_parallel_fit_gives_the_same_forest():
    X, y = _cls_data(n=300)
    a = RandomForest(n_estimators=6, random_state=3).fit(X, y)
    b = RandomForest(n_estimators=6, random_state=3, n_jobs=2).fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_forest_outputs_and_oob():
    X, y = _cls_data()
    labels = np.where(y == 1, "premium", "standard")
    forest = RandomForest(n_estimators=20, oob_score=True, random_state=0).fit(X, labels)
    proba = forest.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(forest.predict(X)) <= {"premium", "standard"}
    assert 0.5 < forest.oob_score_ < 1.0
    assert len(forest.oob_rows_) > 0.95 * len(y)
    staged = list(forest.staged_predict(X))
    assert len(staged) == 20 and (staged[-1] == forest.predict(X)).all()
    assert forest.feature_importances_.sum() == pytest.approx(1.0)


def test_bootstrap_missing_a_rare_class_is_handled():
    X, y = _cls_data(n=200)
    y = y.copy()
    y[:2] = 2                                   # a class with two rows
    forest = RandomForest(n_estimators=15, random_state=1).fit(X, y)
    assert forest.predict_proba(X).shape == (200, 3)
    assert np.allclose(forest.predict_proba(X).sum(axis=1), 1.0)


def test_boosting_reduces_training_loss_and_tracks_sklearn():
    X, y = _reg_data()
    gb = GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=3).fit(X[:400], y[:400])
    assert all(b <= a + 1e-12 for a, b in zip(gb.train_loss_, gb.train_loss_[1:]))
    staged = list(gb.staged_predict(X[400:]))
    assert np.allclose(staged[-1], gb.predict(X[400:]))
    ours = ev.rmse(y[400:], gb.predict(X[400:]))
    sk = SkGB(n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0).fit(X[:400], y[:400])
    assert abs(ours - ev.rmse(y[400:], sk.predict(X[400:]))) < 0.05 * ours
    assert ev.rmse(y[400:], staged[-1]) < ev.rmse(y[400:], staged[4])


def test_boosting_leaf_regularisation_shrinks_steps():
    X, y = _reg_data()
    plain = GradientBoostingRegressor(n_estimators=5, max_depth=3).fit(X, y)
    l2 = GradientBoostingRegressor(n_estimators=5, max_depth=3, leaf_l2=50.0).fit(X, y)
    l1 = GradientBoostingRegressor(n_estimators=5, max_depth=3, leaf_l1=50.0).fit(X, y)
    spread = [np.std(m.predict(X) - m.init_) for m in (plain, l2, l1)]
    assert spread[1] < spread[0] and spread[2] < spread[0]
    with pytest.raises(ValueError):
        GradientBoostingRegressor(learning_rate=0).fit(X, y)
