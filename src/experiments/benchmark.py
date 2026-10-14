from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier

from src import data_prep
from src import evaluate as ev
from src.decision_tree import DecisionTree
from src.svm import PegasosSVM

SEED = data_prep.SEED
TREE_CLS_DEPTH = 12     # chosen from validation (val F1 within 0.003 of depth 15)
TREE_REG_DEPTH = 10     # best validation RMSE
SVM_LAMBDA = 1e-4
N_FOLDS = 5
OUT_DIR = os.path.join("report", "tables")


def load_all():
    """Train/test with the same leakage-safe steps as tree_sweeps.load_splits."""
    df = data_prep.clean(data_prep.load_raw())
    y_all = np.log(df["price"].to_numpy(dtype=float))
    itr, iva, ite = data_prep.split_indices(y_all, seed=SEED)
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[itr] = True
    X, y, _ = data_prep.make_features(df, fit_mask=fit_mask)
    Xtr, Xte = data_prep.impute_median(X[itr], X[ite])
    ytr, yte = y[itr], y[ite]
    tier_tr, thr = data_prep.make_tier_label(ytr)
    tier_te, _ = data_prep.make_tier_label(yte, threshold=thr)
    return Xtr, ytr, tier_tr, Xte, yte, tier_te


def kfold_indices(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def cv_eval(make_model, X, y, metric_fn, scale=False):
    """5-fold CV on TRAIN. Returns metric list, fit times, predict times."""
    folds = kfold_indices(len(X), N_FOLDS, SEED)
    scores, fit_t, pred_t = [], [], []
    for i in range(N_FOLDS):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != i])
        Xa, Xb = X[tr], X[va]
        if scale:                      # scaler fit on the fold's train part only
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
    Xtr, ytr, ctr, Xte, yte, cte = load_all()
    print("train", Xtr.shape, "test", Xte.shape)

    rows = []

    def record(task, name, metric, scores, fit_t, pred_t):
        rows.append(dict(task=task, model=name, metric=metric,
                         cv_mean=np.mean(scores), cv_std=np.std(scores),
                         fit_s=np.mean(fit_t), predict_s=np.mean(pred_t)))
        print(f"{task} {name:22s} {metric}={np.mean(scores):.4f} +/- {np.std(scores):.4f} "
              f"fit={np.mean(fit_t):.2f}s pred={np.mean(pred_t):.3f}s")

    # ---- Task A: regression (log price), RMSE ----
    s = cv_eval(lambda: DecisionTree(task="regression", criterion="mse",
                                     max_depth=TREE_REG_DEPTH), Xtr, ytr, ev.rmse)
    record("A", "ours tree", "rmse", *s)
    s = cv_eval(lambda: DecisionTreeRegressor(max_depth=TREE_REG_DEPTH, random_state=SEED),
                Xtr, ytr, ev.rmse)
    record("A", "sklearn tree", "rmse", *s)

    # ---- Task B: classification (price tier), F1 ----
    s = cv_eval(lambda: DecisionTree(task="classification", criterion="gini",
                                     max_depth=TREE_CLS_DEPTH), Xtr, ctr, f1_metric)
    record("B", "ours tree", "f1", *s)
    s = cv_eval(lambda: DecisionTreeClassifier(criterion="gini", max_depth=TREE_CLS_DEPTH,
                                               random_state=SEED), Xtr, ctr, f1_metric)
    record("B", "sklearn tree", "f1", *s)

    s = cv_eval(lambda: PegasosSVM(lambda_=SVM_LAMBDA, random_state=SEED),
                Xtr, ctr, f1_metric, scale=True)
    record("B", "ours pegasos svm", "f1", *s)
    # sklearn SGDClassifier(hinge) is the matched solver; alpha plays the role of lambda
    s = cv_eval(lambda: SGDClassifier(loss="hinge", alpha=SVM_LAMBDA, random_state=SEED),
                Xtr, ctr, f1_metric, scale=True)
    record("B", "sklearn sgd (hinge)", "f1", *s)

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "benchmark_cv.csv"), index=False)

    # ---- Single test-set evaluation, final configs, fit on all TRAIN ----
    print("\nTEST SET (one evaluation, configs fixed from validation):")
    Xs_tr, Xs_te = data_prep.standardize(Xtr, Xte)
    test = []
    for name, m, Xa, Xb, ya, yb, kind in [
        ("ours tree reg", DecisionTree(task="regression", criterion="mse", max_depth=TREE_REG_DEPTH),
         Xtr, Xte, ytr, yte, "reg"),
        ("sklearn tree reg", DecisionTreeRegressor(max_depth=TREE_REG_DEPTH, random_state=SEED),
         Xtr, Xte, ytr, yte, "reg"),
        ("ours tree cls", DecisionTree(task="classification", criterion="gini", max_depth=TREE_CLS_DEPTH),
         Xtr, Xte, ctr, cte, "cls"),
        ("sklearn tree cls", DecisionTreeClassifier(max_depth=TREE_CLS_DEPTH, random_state=SEED),
         Xtr, Xte, ctr, cte, "cls"),
        ("ours svm", PegasosSVM(lambda_=SVM_LAMBDA, random_state=SEED), Xs_tr, Xs_te, ctr, cte, "svm"),
        ("sklearn sgd", SGDClassifier(loss="hinge", alpha=SVM_LAMBDA, random_state=SEED),
         Xs_tr, Xs_te, ctr, cte, "svm"),
    ]:
        m.fit(Xa, ya)
        p = m.predict(Xb)
        if kind == "reg":
            r = dict(model=name, rmse=ev.rmse(yb, p), mae=ev.mae(yb, p), r2=ev.r2(yb, p))
        else:
            sc = m.decision_function(Xb) if kind == "svm" else m.predict_proba(Xb)[:, 1]
            pr, rc, f1 = ev.precision_recall_f1(yb, p)
            r = dict(model=name, precision=pr, recall=rc, f1=f1, auc=ev.roc_auc(yb, sc))
        test.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    pd.DataFrame(test).to_csv(os.path.join(OUT_DIR, "benchmark_test.csv"), index=False)


if __name__ == "__main__":
    main()