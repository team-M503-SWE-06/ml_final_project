"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator should:
  1. load + clean the data,
  2. build features, make the splits, standardize (for the SVM),
  3. train YOUR decision tree (Task A regression, Task B classification),
  4. train YOUR Pegasos SVM (Task B classification),
  5. train the scikit-learn baselines with matched settings,
  6. compute metrics and SAVE all figures to report/figures/,
  7. print a final comparison table.

Everything must be deterministic (fixed seeds). Keep print output tidy so a
grader can read the headline numbers straight from the console.
"""

from __future__ import annotations

from . import data_prep
# from .decision_tree import DecisionTree
# from .svm import PegasosSVM
# from . import evaluate


def main() -> None:
    # 1. data -------------------------------------------------------------
    # df = data_prep.load_raw()
    # df = data_prep.clean(df)
    # X, y, feat_names = data_prep.make_features(df)
    # splits = data_prep.train_val_test_split(X, y)
    # ...

    # 2. Task A: regression (price) --------------------------------------
    #    - your DecisionTree(task="regression", criterion="mse")
    #    - sklearn DecisionTreeRegressor baseline

    # 3. Task B: classification (price tier) -----------------------------
    #    - make_tier_label using TRAIN median
    #    - your DecisionTree(task="classification")
    #    - your PegasosSVM (standardized features)
    #    - sklearn baselines (DecisionTreeClassifier, SVC/LinearSVC)

    # 4. metrics + figures + comparison ----------------------------------
    #    evaluate.compare({...}); save plots to report/figures/

    raise NotImplementedError("TODO: wire up the full pipeline")


if __name__ == "__main__":
    main()
