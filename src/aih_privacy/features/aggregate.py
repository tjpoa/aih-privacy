# src/aih_privacy/features/aggregate.py
from __future__ import annotations
import pandas as pd


DEFAULT_TRIAL_FEATURES = [
    "c2_max_max", "c2_max_mean", "c2_max_std",
    "c8_max", "c8_mean",
    "c9_mean",
    "c1_max_max",
    "n_windows",
]


def make_trial_df(df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates window-level features into one row per trial.

    Expected columns in df_windows:
      - trial_id, subject_id, label
      - c2_max, c8, c9, c1_max (at least)
    """
    required = {"trial_id", "subject_id", "label", "c2_max", "c8", "c9", "c1_max"}
    missing = required - set(df_windows.columns)
    if missing:
        raise ValueError(f"df_windows missing columns: {sorted(missing)}")

    df_trials = (
        df_windows
        .groupby(["trial_id", "subject_id", "label"], as_index=False)
        .agg(
            c2_max_max=("c2_max", "max"),
            c2_max_mean=("c2_max", "mean"),
            c2_max_std=("c2_max", "std"),
            c8_max=("c8", "max"),
            c8_mean=("c8", "mean"),
            c9_mean=("c9", "mean"),
            c1_max_max=("c1_max", "max"),
            n_windows=("c2_max", "count"),
        )
    )

    # std can be NaN if only one window exists (rare) -> 0
    df_trials["c2_max_std"] = df_trials["c2_max_std"].fillna(0.0)

    return df_trials
