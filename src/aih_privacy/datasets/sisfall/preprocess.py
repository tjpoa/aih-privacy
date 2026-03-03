from scipy.signal import butter, lfilter
import pandas as pd

# PREPROCESSING DATASET UTILS

def butter_lowpass(cutoff_hz: float, fs_hz: float, order: int = 4):
    nyq = 0.5 * fs_hz
    w = cutoff_hz / nyq
    b, a = butter(order, w, btype="low")
    return b, a

def filter_one_group(g: pd.DataFrame, b, a, cols=("ax","ay","az")) -> pd.DataFrame:
    # manter a ordem temporal dentro do trial
    g = g.sort_values("sample_idx").copy()
    for c in cols:
        g[c] = lfilter(b, a, g[c].to_numpy(dtype=float))
    return g

def apply_filter_by_trial(df: pd.DataFrame,
                          cutoff_hz: float = 5.0,
                          fs_hz: float = 200.0,
                          order: int = 4,
                          cols=("ax","ay","az")) -> pd.DataFrame:
    b, a = butter_lowpass(cutoff_hz, fs_hz, order)
    # group_keys=False evita index multi-nível no retorno
    out = (
        df.groupby("trial_id", group_keys=False)
        .apply(filter_one_group, b=b, a=a, cols=cols, include_groups=False)
    )
    return out

def apply_filter_by_trial_v2(df: pd.DataFrame,
                          cutoff_hz: float = 20.0, # CHANGED: Increased from 5.0 to 20.0
                          fs_hz: float = 200.0,
                          order: int = 4,
                          # CHANGED: Added Gyroscope columns if available
                          cols=("ax", "ay", "az", "gx", "gy", "gz")) -> pd.DataFrame:
    
    # 1. Filter Check
    # Only filter columns that actually exist in the dataframe
    valid_cols = [c for c in cols if c in df.columns]
    
    # 2. Filter Design (Butterworth)
    # A 20Hz filter removes sensor electrical noise but keeps the gait 'impacts'
    b, a = butter_lowpass(cutoff_hz, fs_hz, order)

    # 3. Apply
    # group_keys=False prevents the multi-index creation
    out = (
        df.groupby("trial_id", group_keys=False)
        .apply(filter_one_group, b=b, a=a, cols=valid_cols, include_groups=False)
    )
    return out

def rename_axes(df):
    """
    Standardiza os nomes dos eixos de aceleração para ax, ay, az.

    Espera colunas do SisFall:
      - acc_x_adxl345
      - acc_y_adxl345
      - acc_z_adxl345

    Devolve uma cópia do DataFrame com:
      - ax, ay, az
    """
    required = {
        "acc_x_adxl345",
        "acc_y_adxl345",
        "acc_z_adxl345",
        "gyro_x_itg3200", 
        "gyro_y_itg3200", 
        "gyro_z_itg3200"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"rename_axes: colunas em falta: {sorted(missing)}"
        )

    return df.rename(
        columns={
            "acc_x_adxl345": "ax",
            "acc_y_adxl345": "ay",
            "acc_z_adxl345": "az",
            "gyro_x_itg3200": "gx",
            "gyro_y_itg3200": "gy",
            "gyro_z_itg3200": "gz"
        }
    )