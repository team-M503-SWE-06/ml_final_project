import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.decision_tree import DecisionTree

# Tiny example with no tied splits: sklearn builds the same tree for every
# random_state, so the expected tree is fully determined.
TOY_X = np.array([[12, 9], [6, 8], [11, 10], [9, 5], [4, 12], [10, 4],
                  [1, 11], [7, 6], [8, 1], [5, 7], [2, 3], [3, 2]], dtype=float)
TOY_Y_CLS = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
TOY_Y_REG = np.array([23, 11, 8, 18, 21, 13, 12, 12, 10, 12, 15, 10], dtype=float)


def _ours(tree):
    """Pre-order list of (feature, threshold, n_samples, value) for our tree."""
    out = []

    def walk(node):
        feature = None if node.is_leaf() else node.feature
        threshold = None if node.is_leaf() else node.threshold
        out.append((feature, threshold, node.n_samples, np.ravel(node.value)))
        if not node.is_leaf():
            walk(node.left)
            walk(node.right)
    walk(tree.root)
    return out


def _sklearn(model):
    """Same pre-order list for a fitted sklearn tree (its node ids are pre-order)."""
    t = model.tree_
    out = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        value = t.value[i][0]
        if isinstance(model, DecisionTreeClassifier):
            value = value / value.sum()
        out.append((None if leaf else int(t.feature[i]), None if leaf else float(t.threshold[i]),
                    int(t.n_node_samples[i]), np.ravel(value)))
    return out


def _assert_same_tree(ours, theirs):
    assert len(ours) == len(theirs)
    for (f1, t1, n1, v1), (f2, t2, n2, v2) in zip(ours, theirs):
        assert f1 == f2
        assert n1 == n2
        if t1 is not None:
            assert t1 == pytest.approx(t2, abs=1e-6)   # sklearn stores thresholds in float32
        assert np.allclose(v1, v2)


def _determined_sklearn_tree(make_model, X, y):
    """
    Fit sklearn with several random_states and check they all agree. sklearn
    breaks ties between equally good splits at random, so only a tie-free
    example has one correct answer to compare against.
    """
    trees = [_sklearn(make_model(seed).fit(X, y)) for seed in range(10)]
    for other in trees[1:]:
        _assert_same_tree(other, trees[0])
    return trees[0]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_toy_classification_matches_sklearn(criterion):
    expected = _determined_sklearn_tree(
        lambda s: DecisionTreeClassifier(criterion=criterion, random_state=s), TOY_X, TOY_Y_CLS)
    ours = DecisionTree(task="classification", criterion=criterion).fit(TOY_X, TOY_Y_CLS)
    _assert_same_tree(_ours(ours), expected)


def test_toy_regression_matches_sklearn():
    expected = _determined_sklearn_tree(
        lambda s: DecisionTreeRegressor(max_depth=3, random_state=s), TOY_X, TOY_Y_REG)
    ours = DecisionTree(task="regression", criterion="mse", max_depth=3).fit(TOY_X, TOY_Y_REG)
    _assert_same_tree(_ours(ours), expected)


@pytest.mark.parametrize("criterion", ["gini", "entropy", "mse"])
def test_random_data_matches_sklearn(criterion):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5))
    noise = rng.normal(scale=0.3, size=300)
    params = dict(max_depth=4, min_samples_leaf=5)
    if criterion == "mse":
        y = np.sin(X[:, 0]) + X[:, 2] + noise
        ours = DecisionTree(task="regression", criterion="mse", **params)
        make = lambda s: DecisionTreeRegressor(random_state=s, **params)
    else:
        y = (X[:, 0] + 0.5 * X[:, 1] + noise > 0).astype(int)
        ours = DecisionTree(criterion=criterion, **params)
        make = lambda s: DecisionTreeClassifier(criterion=criterion, random_state=s, **params)

    expected = _determined_sklearn_tree(make, X, y)
    _assert_same_tree(_ours(ours.fit(X, y)), expected)
    assert np.allclose(ours.feature_importances_, make(0).fit(X, y).feature_importances_)


def _noisy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] * X[:, 1] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("param, value", [
    ("max_depth", 3),
    ("min_samples_split", 50),
    ("min_samples_leaf", 20),
    ("min_impurity_decrease", 0.01),
])
def test_each_stopping_parameter_changes_the_tree(param, value):
    X, y = _noisy_data()
    full = DecisionTree(criterion="gini").fit(X, y)
    limited = DecisionTree(criterion="gini", **{param: value}).fit(X, y)
    assert limited.get_n_leaves() < full.get_n_leaves()


def test_stopping_parameters_are_respected():
    X, y = _noisy_data()
    tree = DecisionTree(criterion="entropy", max_depth=5, min_samples_leaf=7).fit(X, y)
    assert tree.get_depth() <= 5
    leaves = [n for n in _ours(tree) if n[0] is None]
    assert min(n_samples for _, _, n_samples, _ in leaves) >= 7


def test_predict_proba_and_labels():
    X, y = _noisy_data()
    labels = np.where(y == 1, "premium", "standard")
    tree = DecisionTree(criterion="gini", max_depth=4).fit(X, labels)
    proba = tree.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(tree.classes_) == ["premium", "standard"]
    assert (tree.predict(X) == tree.classes_[proba.argmax(axis=1)]).all()


def test_regression_leaf_is_mean_and_fully_grown_tree_fits_train():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] + rng.normal(size=100)
    stump = DecisionTree(task="regression", criterion="mse", max_depth=0).fit(X, y)
    assert np.allclose(stump.predict(X), y.mean())
    full = DecisionTree(task="regression", criterion="mse").fit(X, y)
    assert np.allclose(full.predict(X), y)


def test_feature_importances_sum_to_one():
    X, y = _noisy_data()
    tree = DecisionTree(criterion="gini", max_depth=6).fit(X, y)
    assert tree.feature_importances_.shape == (X.shape[1],)
    assert tree.feature_importances_.sum() == pytest.approx(1.0)
    assert tree.feature_importances_[:2].sum() > tree.feature_importances_[2:].sum()


def test_invalid_input_is_rejected():
    X, y = _noisy_data()
    with pytest.raises(ValueError):
        DecisionTree(task="classification", criterion="mse").fit(X, y)
    with pytest.raises(ValueError):
        DecisionTree(min_samples_leaf=0).fit(X, y)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        DecisionTree().fit(X_nan, y)
    with pytest.raises(AttributeError):
        DecisionTree(task="regression", criterion="mse").fit(X, y.astype(float)).predict_proba(X)


def test_predict_rejects_input_that_does_not_match_fit():
    X, y = _noisy_data()
    tree = DecisionTree(criterion="gini", max_depth=3).fit(X, y)
    with pytest.raises(ValueError):
        tree.predict(X[:, :2])
    with pytest.raises(ValueError):
        tree.predict(np.column_stack([X, X[:, 0]]))
    with pytest.raises(ValueError):
        tree.predict(X[0])
    X_nan = X[:3].copy()
    X_nan[0, 1] = np.nan
    with pytest.raises(ValueError):
        tree.predict(X_nan)
    assert tree.n_features_in_ == X.shape[1]
    assert tree.predict(np.empty((0, X.shape[1]))).shape == (0,)


def test_huge_values_do_not_overflow_the_threshold():
    X = np.array([[1e308], [1.5e308], [1.6e308], [1.7e308]])
    tree = DecisionTree(criterion="gini").fit(X, [0, 1, 0, 1])
    assert np.isfinite(tree.root.threshold)
    assert (tree.predict(X) == [0, 1, 0, 1]).all()
    with pytest.raises(ValueError):
        DecisionTree().fit(np.array([[-np.inf], [np.inf]]), [0, 1])


def test_y_is_validated():
    X, y = _noisy_data()
    column = DecisionTree(criterion="gini", max_depth=3).fit(X, y[:, None])
    flat = DecisionTree(criterion="gini", max_depth=3).fit(X, y)
    assert (column.predict(X) == flat.predict(X)).all()
    y_reg = X[:, 0].copy()
    y_reg[0] = np.nan
    with pytest.raises(ValueError):
        DecisionTree(task="regression", criterion="mse").fit(X, y_reg)
    with pytest.raises(ValueError):
        DecisionTree().fit(X, np.where(y == 1, np.nan, 0.0))
    with pytest.raises(ValueError):
        DecisionTree().fit(X, np.column_stack([y, y]))


def test_size_parameters_must_be_integers_and_queries_need_fit():
    X, y = _noisy_data()
    with pytest.raises(ValueError):
        DecisionTree(max_depth=2.5).fit(X, y)
    assert DecisionTree(max_depth=np.int64(2)).fit(X, y).get_depth() <= 2
    for query in (DecisionTree().get_depth, DecisionTree().get_n_leaves):
        with pytest.raises(RuntimeError):
            query()
    with pytest.raises(RuntimeError):
        DecisionTree().predict(X)


def test_very_deep_chain_tree_does_not_hit_the_recursion_limit():
    n = 1100                                   # deeper than the ~1000 recursion limit
    X = np.arange(n, dtype=float)[:, None]
    y = np.arange(n) % 2                       # every split peels off one row
    tree = DecisionTree(criterion="gini").fit(X, y)
    expected = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert tree.get_depth() == expected.get_depth() == n - 1
    assert tree.get_n_leaves() == n
    assert (tree.predict(X) == y).all()


def test_constant_large_targets_form_a_pure_node():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.where(X[:, 0] > 0, 123456789.123, 1.5 * 123456789.123)
    tree = DecisionTree(task="regression", criterion="mse").fit(X, y)
    assert tree.get_n_leaves() == 2
    assert tree.feature_importances_[1:].sum() == 0.0


def test_impurity_values():
    gini = DecisionTree(criterion="gini")
    gini.classes_ = np.array([0, 1])
    entropy = DecisionTree(criterion="entropy")
    entropy.classes_ = np.array([0, 1])
    mse = DecisionTree(task="regression", criterion="mse")
    assert gini._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert gini._impurity(np.array([1, 1, 1])) == 0.0
    assert entropy._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert entropy._impurity(np.array([0, 1, 1, 1])) == pytest.approx(0.8112781244591328)
    assert mse._impurity(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.25)
