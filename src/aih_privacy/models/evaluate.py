# src/aih_privacy/models/evaluation.py
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

FEATURES = [
    "c2_max_max","c2_max_mean","c2_max_std",
    "c8_max","c8_mean","c9_mean",
    "c1_max_max",
    # "n_windows"
]

def eval_utility_binary(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    ba = (se + sp) / 2
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"SE": se, "SP": sp, "BA": ba, "F1": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn}

def eval_groupkfold(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    cms = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        model.fit(X[tr], y[tr])
        y_pred = model.predict(X[te])

        tn, fp, fn, tp = confusion_matrix(y[te], y_pred).ravel()

        se = tp / (tp + fn) if (tp + fn) else 0.0
        sp = tn / (tn + fp) if (tn + fp) else 0.0
        ba = balanced_accuracy_score(y[te], y_pred)
        f1 = f1_score(y[te], y_pred)

        rows.append({
            "fold": fold,
            "SE": se,
            "SP": sp,
            "BA": ba,
            "F1": f1,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })
        cms.append((tn, fp, fn, tp))

    return pd.DataFrame(rows), np.sum(cms, axis=0)


# def run_groupkfold_predictions(
#     X: np.ndarray,
#     y: np.ndarray,
#     groups: np.ndarray,
#     fit_predict_fn,
#     n_splits: int = 5,
# ) -> tuple[pd.DataFrame, np.ndarray]:
#     """
#     Generic GroupKFold evaluator.
#     fit_predict_fn: (X_train, y_train, X_test) -> y_pred_test
#     """
#     gkf = GroupKFold(n_splits=n_splits)
#     rows = []
#     cm = np.zeros((2, 2), dtype=int)

#     for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
#         yhat = fit_predict_fn(X[tr], y[tr], X[te])
#         r = eval_binary(y[te], yhat)
#         r["fold"] = fold
#         rows.append(r)
#         cm += confusion_matrix(y[te], yhat, labels=[0, 1])

#     return pd.DataFrame(rows), cm

# def run_lr_groupkfold(df: pd.DataFrame, feature_cols=FEATURES, n_splits=5, random_state=0):
#     X = df[feature_cols].to_numpy(float)
#     y = df["label"].to_numpy(int)
#     groups = df["subject_id"].to_numpy()

#     gkf = GroupKFold(n_splits=n_splits)

#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("lr", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=random_state))
#     ])

#     rows = []
#     cm = np.zeros((2,2), dtype=int)

#     for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
#         clf.fit(X[tr], y[tr])
#         yhat = clf.predict(X[te])

#         r = eval_binary(y[te], yhat)
#         r["fold"] = fold
#         rows.append(r)
#         cm += confusion_matrix(y[te], yhat, labels=[0,1])

#     return pd.DataFrame(rows), cm

# def run_lr_cv(df_windows, feature_cols=("c2_max","c8","c9","c1_max"), n_splits=5):
#     X = df_windows[list(feature_cols)].to_numpy(float)
#     y = df_windows["label"].to_numpy(int)
#     groups = df_windows["subject_id"].to_numpy()

#     gkf = GroupKFold(n_splits=n_splits)

#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
#     ])

#     rows = []
#     cm = np.zeros((2,2), dtype=int)

#     for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
#         clf.fit(X[tr], y[tr])
#         yhat = clf.predict(X[te])

#         r = eval_binary(y[te], yhat)
#         r["fold"] = fold
#         rows.append(r)

#         cm += confusion_matrix(y[te], yhat, labels=[0,1])

#     return pd.DataFrame(rows), cm