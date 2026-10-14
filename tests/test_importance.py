import numpy as np
from src import evaluate as ev
from src.decision_tree import DecisionTree
from src.importance import one_hot_groups, permutation_importance


def test_one_hot_groups():
    names = ["area", "district=A", "lat", "district=other", "category=flat"]
    assert one_hot_groups(names) == {"area": [0], "district": [1, 3], "lat": [2], "category": [4]}


def _data():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=600)
    noise = rng.normal(size=600)
    level = rng.integers(0, 3, size=600)
    one_hot = np.eye(3)[level]
    X = np.column_stack([signal, noise, one_hot])
    y = 2 * signal + level + rng.normal(scale=0.1, size=600)
    names = ["signal", "noise", "cat=a", "cat=b", "cat=c"]
    return X, y, names


def test_signal_beats_noise_and_lower_is_better_sign():
    X, y, names = _data()
    model = DecisionTree(task="regression", criterion="mse", max_depth=6).fit(X[:400], y[:400])
    imp = permutation_importance(model, X[400:], y[400:], ev.rmse, one_hot_groups(names),
                                 higher_is_better=False)
    assert list(imp)[0] == "signal"
    assert imp["signal"][0] > 0.5
    assert imp["cat"][0] > 0.1
    assert abs(imp["noise"][0]) < 0.05


def test_group_shuffle_keeps_one_hot_rows_valid():
    X, y, names = _data()
    seen = []

    class Spy:
        def predict(self, X_in):
            seen.append(X_in[:, 2:].sum(axis=1))
            return np.zeros(len(X_in))

    permutation_importance(Spy(), X, y, ev.rmse, one_hot_groups(names), n_repeats=2)
    assert all(np.allclose(s, 1.0) for s in seen)


def test_deterministic_for_a_seed():
    X, y, names = _data()
    model = DecisionTree(task="regression", criterion="mse", max_depth=4).fit(X, y)
    a = permutation_importance(model, X, y, ev.rmse, one_hot_groups(names), False, seed=1)
    b = permutation_importance(model, X, y, ev.rmse, one_hot_groups(names), False, seed=1)
    assert a == b
