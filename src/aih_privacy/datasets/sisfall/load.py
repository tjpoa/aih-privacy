# src/aih_privacy/datasets/sisfall/load.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from src.aih_privacy.config import DATA_RAW_DIR

FILENAME_PATTERN = re.compile(r"([DF]\d{2})_((?:SA|SE)\d{2})_R(\d{2})\.txt")

COLUMN_NAMES = [
    "acc_x_adxl345", "acc_y_adxl345", "acc_z_adxl345",
    "gyro_x_itg3200", "gyro_y_itg3200", "gyro_z_itg3200",
    "acc_x_mma8451q", "acc_y_mma8451q", "acc_z_mma8451q",
]

SAMPLING_RATE = 200

FACTOR_ADXL345 = 32 / (2**13)
FACTOR_ITG3200 = 4000 / (2**16)
FACTOR_MMA8451Q = 16 / (2**14)

def default_sisfall_dir() -> Path:
    """Default raw directory for SisFall based on config.py."""
    return DATA_RAW_DIR / "sisfall"

def parse_filename(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from SisFall filename.

    Returns dict with:
      - activity_code (D01/F03...)
      - subject_id (SA06/SE14...)
      - age_group (SA/SE)
      - rep (int)
      - label (0=ADL, 1=Fall)
      - trial_id (e.g., D01_SA01_R01)
    """
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None

    activity_code = m.group(1)
    subject_id = m.group(2)
    rep = int(m.group(3))
    age_group = subject_id[:2]
    label = 1 if activity_code.startswith("F") else 0
    trial_id = f"{activity_code}_{subject_id}_R{rep:02d}"

    return {
        "activity_code": activity_code,
        "subject_id": subject_id,
        "age_group": age_group,
        "rep": rep,
        "label": label,
        "trial_id": trial_id,
    }


def iter_files(root: Path) -> list[Path]:
    """Returns a sorted list of SisFall .txt files matching the expected pattern."""
    return sorted(
        f for f in root.rglob("*.txt")
        if FILENAME_PATTERN.match(f.name)
    )


def load_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single SisFall txt file into a DataFrame (9 cols), applying scaling factors.
    """
    df = pd.read_csv(
        filepath,
        header=None,
        sep=r"[,\s;]+",
        engine="python",
        usecols=range(9),
    )
    df.columns = COLUMN_NAMES

    # Apply conversion factors
    df[["acc_x_adxl345", "acc_y_adxl345", "acc_z_adxl345"]] *= FACTOR_ADXL345
    df[["gyro_x_itg3200", "gyro_y_itg3200", "gyro_z_itg3200"]] *= FACTOR_ITG3200
    df[["acc_x_mma8451q", "acc_y_mma8451q", "acc_z_mma8451q"]] *= FACTOR_MMA8451Q

    return df


def build_sisfall_df(
    root: Optional[Path] = None,
    fs_hz: float = SAMPLING_RATE,
    add_time: bool = True,
    to_category: bool = True,
) -> pd.DataFrame:
    """
    Loads all SisFall files under root into a single DataFrame.

    Adds:
      - activity_code, subject_id, age_group, label, rep, trial_id
      - sample_idx
      - time_s (if add_time=True)
      - source_file
    """
    if root is None:
        root = default_sisfall_dir()

    files = iter_files(root)
    if not files:
        raise FileNotFoundError(f"No SisFall files found under: {root}")

    all_dfs = []
    for path in files:
        meta = parse_filename(path)
        if meta is None:
            continue

        df = load_file(path)
        df = df.reset_index(drop=True)

        df["sample_idx"] = np.arange(len(df), dtype=np.int32)
        if add_time:
            df["time_s"] = df["sample_idx"] / float(fs_hz)

        for k, v in meta.items():
            df[k] = v

        df["source_file"] = path.name

        if to_category:
            for c in ["trial_id", "activity_code", "subject_id", "age_group"]:
                df[c] = df[c].astype("category")

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid SisFall files matched the expected naming pattern.")

    return pd.concat(all_dfs, ignore_index=True)
