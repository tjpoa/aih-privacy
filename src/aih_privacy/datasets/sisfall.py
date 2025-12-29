from pathlib import Path
import re
import pandas as pd
import numpy as np

from src.aih_privacy.config import DATA_RAW_DIR

SISFALL_DIR = DATA_RAW_DIR / "sisfall"

SAMPLING_RATE = 200

COLUMN_NAMES = [
    "acc_x_adxl345", "acc_y_adxl345", "acc_z_adxl345",
    "gyro_x_itg3200", "gyro_y_itg3200", "gyro_z_itg3200",
    "acc_x_mma8451q", "acc_y_mma8451q", "acc_z_mma8451q",
]



FACTOR_ADXL345 = 32 / (2**13)
FACTOR_ITG3200 = 4000 / (2**16)
FACTOR_MMA8451Q = 16 / (2**14)

WINDOW_SIZE = 200  # 1 second  (SisFall = 200 Hz)



def load_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        header=None,
        sep=r"[,\s;]+",
        engine="python",
        usecols=range(9),
    )

    df.columns = COLUMN_NAMES

    df[["acc_x_adxl345", "acc_y_adxl345", "acc_z_adxl345"]] *= FACTOR_ADXL345
    df[["gyro_x_itg3200", "gyro_y_itg3200", "gyro_z_itg3200"]] *= FACTOR_ITG3200
    df[["acc_x_mma8451q", "acc_y_mma8451q", "acc_z_mma8451q"]] *= FACTOR_MMA8451Q

    return df

pattern = re.compile(r"([DF]\d{2})_((?:SA|SE)\d{2})_R(\d{2})\.txt")

def parse_filename(path: Path):
    """
    Extract metadata from SisFall filename.
    Returns: (activity_code, subject_id, age_group, label) or None.
    """
    match = pattern.match(path.name)
    if not match:
        return None

    activity_code = match.group(1)   # e.g. D01, F03
    subject_id    = match.group(2)   # e.g. SA06, SE14
    age_group     = subject_id[:2]   # SA or SE
    label         = 1 if activity_code.startswith("F") else 0

    return activity_code, subject_id, age_group, label


def iter_files():
    """
    Returns a sorted list of SisFall .txt files matching the expected pattern.
    """
    # print(SISFALL_DIR)

    return sorted(
        f for f in SISFALL_DIR.rglob("*.txt")
        if pattern.match(f.name)
    )

def acc_magnitude(df):
    """
    Compute acc magnitude from gx, gy, gz columns
    """
    return np.sqrt(
        df["acc_x_adxl345"]**2 +
        df["acc_y_adxl345"]**2 +
        df["acc_z_adxl345"]**2
    )

def gyro_magnitude(df):
    """
    Compute gyroscope magnitude from gx, gy, gz columns
    """
    return np.sqrt(
        df["gyro_x_itg3200"]**2 +
        df["gyro_y_itg3200"]**2 +
        df["gyro_z_itg3200"]**2
    ).values



def window_stat(signal, stat_fn):
    return [
        stat_fn(signal[i:i+WINDOW_SIZE])
        for i in range(0, len(signal) - WINDOW_SIZE, WINDOW_SIZE)
    ]

def sliding_windows(signal, window_size, step):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start:start + window_size])
    return windows


def extract_features(window):
    return {
        "max": np.max(window),
        "mean": np.mean(window),
        "std": np.std(window),
        "range": np.max(window) - np.min(window),
        "energy": np.sum(window ** 2),
    }
