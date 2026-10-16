"""benchmark.py: Dev 4. OUR models vs scikit-learn, matched settings.

Run from repo root:  python -m src.experiments.benchmark
Configs are fixed BEFORE the test run:
  - trees: from a 5-fold CV search on TRAIN (Camalzadeh) + our depth sweeps
  - SVM lambda: best linear lambda on VALIDATION (svm_sweeps.py)
5-fold CV uses TRAIN only; the test set is touched once, at the end.
Tree rows are written first, so the script still works if the SVM is not merged yet.
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC

from src import data_prep
from src import evaluate as ev
from src.decision_tree import DecisionTree

SEED = data_prep.SEED
TREE_REG_PARAMS = dict(min_samples_split=75)                       # no depth limit
TREE_CLS_PARAMS = dict(criterion="gini", max_depth=20, min_impurity_decrease=2e-5)
SVM_LAMBDA = 1e-3
SVC_SUBSAMPLE = 10_000          # SVC is O(n^2): RBF SVC runs on a train subsample
N_FOLDS = 5
OUT_DIR = os.path.join("report", "tables")


def load_all():
    """Train/val/test with the same leakage-safe steps as tree_sweeps.load_splits."""
    df = data_prep.clean(data_prep.load_raw())
    y_all = np.log(df["price"].to_numpy(dtype=float))
    itr, iva, ite = data_prep.split_indices(y_all, seed=SEED)
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[itr] = True
    X, y, _ = data_prep.make_features(df, fit_mask=fit_mask)
    Xtr, Xte, Xva = data_prep.impute_median(X[itr], X[ite], X[iva])
    ytr, yte, yva = y[itr], y[ite], y[iva]
    tier_tr, thr = data_prep.make_tier_label(ytr)
    tier_te, _ = data_prep.make_tier_label(yte, threshold=thr)
    return Xtr, ytr, tier_tr, Xte, yte, tier_te, Xva, yva


def svm_available() -> bool:
    """True if src/svm.py is implemented (it may still be the starter stub)."""
    try:
        from src.svm import PegasosSVM
        rng = np.random.default_rng(0)
        PegasosSVM(lambda_=SVM_LAMBDA, random_state=SEED).fit(
            rng.normal(size=(20, 3)), np.tile([0, 1], 10))
        return True
    except (NotImplementedError, ImportError):
        return False


def kfold_indices(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def cv_eval(make_model, X, y, metric_fn, scale=False):
    """5-fold CV on TRAIN. scale=True standardizes with the fold's train stats."""
    folds = kfold_indices(len(X), N_FOLDS, SEED)
    scores, fit_t, pred_t = [], [], []
    for i in range(N_FOLDS):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != i])
        Xa, Xb = X[tr], X[va]
        if scale:
            Xa, Xb = data_prep.standardize(Xa, Xb)
        model = make_model()
        _, ft = timed(lambda: model.fit(Xa, y[tr]))
        pred, pt = timed(lambda: model.predict(Xb))
        scores.append(metric_fn(y[va], pred))
        fit_t.append(ft)
        pred_t.append(pt)
    return scores, fit_t, pred_t


def f1_metric(yt, yp):
    return ev.precision_recall_f1(yt, yp)[2]


def main():
    Xtr, ytr, ctr, Xte, yte, cte, Xva, yva = load_all()
    print("train", Xtr.shape, "test", Xte.shape)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    def record(task, name, metric, res):
        scores, fit_t, pred_t = res
        rows.append(dict(task=task, model=name, metric=metric,
                         cv_mean=np.mean(scores), cv_std=np.std(scores),
                         fit_s=np.mean(fit_t), predict_s=np.mean(pred_t)))
        print(f"{task} {name:26s} {metric}={np.mean(scores):.4f} +/- {np.std(scores):.4f} "
              f"fit={np.mean(fit_t):.2f}s pred={np.mean(pred_t):.3f}s")
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "benchmark_cv.csv"), index=False)

    # ---- trees (written first) ----
    record("A", "ours tree", "rmse", cv_eval(
        lambda: DecisionTree(task="regression", criterion="mse", **TREE_REG_PARAMS),
        Xtr, ytr, ev.rmse))
    record("A", "sklearn tree", "rmse", cv_eval(
        lambda: DecisionTreeRegressor(random_state=SEED, **TREE_REG_PARAMS),
        Xtr, ytr, ev.rmse))
    record("B", "ours tree", "f1", cv_eval(
        lambda: DecisionTree(task="classification", **TREE_CLS_PARAMS), Xtr, ctr, f1_metric))
    record("B", "sklearn tree", "f1", cv_eval(
        lambda: DecisionTreeClassifier(random_state=SEED, **TREE_CLS_PARAMS),
        Xtr, ctr, f1_metric))

    # ---- SVMs (only if src/svm.py is implemented) ----
    has_svm = svm_available()
    n_fold_train = int(len(Xtr) * (N_FOLDS - 1) / N_FOLDS)
    if has_svm:
        from src.svm import PegasosSVM
        record("B", "ours pegasos svm", "f1", cv_eval(
            lambda: PegasosSVM(lambda_=SVM_LAMBDA, random_state=SEED),
            Xtr, ctr, f1_metric, scale=True))
    else:
        print("src/svm.py is not implemented yet: skipping our SVM rows.")
    # sklearn baselines: scaler INSIDE a Pipeline (fit on each fold's train part)
    record("B", "sklearn sgd (hinge)", "f1", cv_eval(
        lambda: make_pipeline(StandardScaler(),
                              SGDClassifier(loss="hinge", alpha=SVM_LAMBDA, random_state=SEED)),
        Xtr, ctr, f1_metric))
    record("B", "sklearn linearsvc", "f1", cv_eval(
        lambda: make_pipeline(StandardScaler(),
                              LinearSVC(loss="hinge", C=1.0 / (SVM_LAMBDA * n_fold_train),
                                        dual=True, max_iter=50_000, random_state=SEED)),
        Xtr, ctr, f1_metric))
    sub = np.random.default_rng(SEED).choice(len(Xtr), SVC_SUBSAMPLE, replace=False)
    record("B", f"sklearn rbf svc ({SVC_SUBSAMPLE} rows)", "f1", cv_eval(
        lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        Xtr[sub], ctr[sub], f1_metric))

    # ---- single test-set evaluation, configs fixed above ----
    print("\nTEST SET (one evaluation, configs fixed before this run):")
    Xs_tr, Xs_te = data_prep.standardize(Xtr, Xte)
    ours_reg = DecisionTree(task="regression", criterion="mse", **TREE_REG_PARAMS).fit(Xtr, ytr)
    sk_reg = DecisionTreeRegressor(random_state=SEED, **TREE_REG_PARAMS).fit(Xtr, ytr)
    ours_cls = DecisionTree(task="classification", **TREE_CLS_PARAMS).fit(Xtr, ctr)
    sk_cls = DecisionTreeClassifier(random_state=SEED, **TREE_CLS_PARAMS).fit(Xtr, ctr)

    test = []
    for name, m in [("ours tree reg", ours_reg), ("sklearn tree reg", sk_reg)]:
        p = m.predict(Xte)
        # Duan smearing: exp(mean log) is biased low, so rescale by the mean
        # exp-residual measured on VALIDATION (train residuals of a deep tree are ~0)
        smear = float(np.mean(np.exp(yva - m.predict(Xva))))
        pa = np.exp(p) * smear
        r = dict(model=name, rmse_log=ev.rmse(yte, p), mae_log=ev.mae(yte, p), r2_log=ev.r2(yte, p),
                 smearing_factor=smear,
                 rmse_azn=ev.rmse(np.exp(yte), pa), mae_azn=ev.mae(np.exp(yte), pa),
                 rmse_azn_uncorrected=ev.rmse(np.exp(yte), np.exp(p)),
                 mae_azn_uncorrected=ev.mae(np.exp(yte), np.exp(p)))
        test.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    sgd = make_pipeline(StandardScaler(), SGDClassifier(
        loss="hinge", alpha=SVM_LAMBDA, random_state=SEED)).fit(Xtr, ctr)
    lsvc = make_pipeline(StandardScaler(), LinearSVC(
        loss="hinge", C=1.0 / (SVM_LAMBDA * len(Xtr)), dual=True,
        max_iter=50_000, random_state=SEED)).fit(Xtr, ctr)
    cls_models = [("ours tree cls", ours_cls, Xte, "proba"),
                  ("sklearn tree cls", sk_cls, Xte, "proba"),
                  ("sklearn sgd", sgd, Xte, "dec"),
                  ("sklearn linearsvc", lsvc, Xte, "dec")]
    if has_svm:
        from src.svm import PegasosSVM
        ours_svm = PegasosSVM(lambda_=SVM_LAMBDA, random_state=SEED).fit(Xs_tr, ctr)
        cls_models.insert(2, ("ours svm", ours_svm, Xs_te, "dec"))
    for name, m, Xb, kind in cls_models:
        p = m.predict(Xb)
        sc = m.predict_proba(Xb)[:, 1] if kind == "proba" else m.decision_function(Xb)
        pr, rc, f1 = ev.precision_recall_f1(cte, p)
        r = dict(model=name, precision=pr, recall=rc, f1=f1, auc=ev.roc_auc(cte, sc))
        test.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    pd.DataFrame(test).to_csv(os.path.join(OUT_DIR, "benchmark_test.csv"), index=False)

    # ---- Problem 5(a): structure / prediction agreement with sklearn ----
    print("\nSTRUCTURE vs scikit-learn (test set):")
    struct = []
    for task, o, s in [("A reg", ours_reg, sk_reg), ("B cls", ours_cls, sk_cls)]:
        agree = float(np.mean(np.isclose(o.predict(Xte), s.predict(Xte))))
        r = dict(task=task, ours_depth=o.get_depth(), sk_depth=s.get_depth(),
                 ours_leaves=o.get_n_leaves(), sk_leaves=s.get_n_leaves(),
                 identical_predictions=agree)
        struct.append(r)
        print(r)
    pd.DataFrame(struct).to_csv(os.path.join(OUT_DIR, "benchmark_structure.csv"), index=False)


if __name__ == "__main__":
    main()