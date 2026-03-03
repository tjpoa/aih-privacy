# src/aih_privacy/privacy/dp_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.aih_privacy.models.evaluate import eval_binary
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix


def clip_X(X: np.ndarray, C: float) -> np.ndarray:
    return np.clip(X, -C, C)

def compute_feature_clips(df, cols, q_low=0.005, q_high=0.995):
    clip_min = {c: float(df[c].quantile(q_low)) for c in cols}
    clip_max = {c: float(df[c].quantile(q_high)) for c in cols}
    return clip_min, clip_max

def dp_noise_laplace(X: np.ndarray, epsilon: float, C: float, rng: np.random.Generator) -> np.ndarray:
    """
    Laplace mechanism on already clipped data.
    Simplified per-feature scale = C/epsilon.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = C / epsilon
    return X + rng.laplace(loc=0.0, scale=scale, size=X.shape)

def dp_laplace_on_features_df(X: pd.DataFrame, epsilon: float, clip_min: dict, clip_max: dict, rng):
    Xp = X.copy()
    for c in Xp.columns:
        a, b = float(clip_min[c]), float(clip_max[c])
        Xp[c] = Xp[c].clip(a, b)

        # Sensibilidade L1 por feature ~ (b-a); Laplace scale = sensitivity/epsilon
        scale = (b - a) / float(epsilon)
        Xp[c] = Xp[c].astype(float) + rng.laplace(0.0, scale, size=len(Xp))
    return Xp

def run_lr_groupkfold_dp_features(
    df_trials: pd.DataFrame,
    feature_cols: list[str],
    eps: float,
    C: float = 3.0,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Per fold:
      - fit StandardScaler on train
      - transform train/test
      - clip both
      - add Laplace noise to TRAIN only
      - fit LR, evaluate on clean test
    """
    X0 = df_trials[feature_cols].to_numpy(float)
    y = df_trials["label"].to_numpy(int)
    groups = df_trials["subject_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    rng_master = np.random.default_rng(seed)

    rows = []
    cm = np.zeros((2, 2), dtype=int)

    for fold, (tr, te) in enumerate(gkf.split(X0, y, groups), 1):
        Xtr0, Xte0 = X0[tr], X0[te]
        ytr, yte = y[tr], y[te]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr0)
        Xte = scaler.transform(Xte0)

        Xtr = clip_X(Xtr, C)
        Xte = clip_X(Xte, C)

        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        Xtr_dp = dp_noise_laplace(Xtr, epsilon=eps, C=C, rng=rng)

        lr = LogisticRegression(max_iter=4000, class_weight="balanced")
        lr.fit(Xtr_dp, ytr)

        yhat = lr.predict(Xte)
        r = eval_binary(yte, yhat)
        r["fold"] = fold
        r["epsilon"] = eps
        r["C"] = C
        rows.append(r)

        cm += confusion_matrix(yte, yhat, labels=[0, 1])

    return pd.DataFrame(rows), cm
