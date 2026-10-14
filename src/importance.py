"""importance.py: permutation feature importance (NumPy only, works with any fitted model).

Impurity-based importance (DecisionTree.feature_importances_, "MDI") is
computed on the TRAINING data and favours features with many distinct
values: in our experiments a pure-noise continuous column entered the MDI
top 8. Permutation importance measures what the model actually loses on
held-out data when one feature's values are shuffled, which breaks its link
with the target while keeping its distribution.

One-hot columns of the same categorical (e.g. every "district=..." column)
are shuffled TOGETHER as one group, so each row still has exactly one level.
"""
from __future__ import annotations

import numpy as np


def one_hot_groups(feature_names) -> dict:
    """
    Map each feature to its column indices: "district=Nasimi" and
    "district=other" form one group "district"; every other column is its
    own group. Order follows the first appearance in feature_names.
    """
    groups = {}
    for j, name in enumerate(feature_names):
        key = name.split("=", 1)[0] if "=" in name else name
        groups.setdefault(key, []).append(j)
    return groups


def permutation_importance(model, X, y, metric, groups: dict, higher_is_better: bool = True,
                           n_repeats: int = 5, seed: int = 42):
    """
    Score drop when each group of columns is shuffled (rows permuted jointly).

    metric(y_true, y_pred) -> float is computed on model.predict(X). The
    importance of a group is baseline - shuffled score (or shuffled -
    baseline when lower is better), so a positive value always means "the
    model needs this feature". Returns {group: (mean, std)} over n_repeats,
    sorted from most to least important.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    sign = 1.0 if higher_is_better else -1.0
    baseline = metric(y, model.predict(X))

    result = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(sign * (baseline - metric(y, model.predict(X_shuffled))))
        result[name] = (float(np.mean(drops)), float(np.std(drops)))
    return dict(sorted(result.items(), key=lambda item: -item[1][0]))
