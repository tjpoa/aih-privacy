import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, iqr

from src.aih_privacy.features.windowing import sliding_windows_indices

def magnitude(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def compute_c_features_from_axes(ax, ay, az):
    # "C1" e "C2" como no teu código atual
    c1 = np.sqrt(ax**2 + ay**2 + az**2)
    c2 = np.sqrt(ax**2 + az**2)  # (no teu pipeline)
    return c1, c2

def extract_windows_features_one_trial(g: pd.DataFrame, fs: float, win_sec: float, step_sec: float):
    """
    Extrai features por janela para um único trial (g).
    Assume que g já está filtrado (df_f) e ordenado por sample_idx.
    """
    g = g.sort_values("sample_idx")
    win_size = int(round(win_sec * fs))
    step = int(round(step_sec * fs))

    ax = g["ax"].to_numpy(float)
    ay = g["ay"].to_numpy(float)
    az = g["az"].to_numpy(float)

    # label é constante por trial no SisFall (Fxx ou Dxx), mas fazemos robusto:
    y = g["label"].to_numpy(int)

    # C2 e C1 por amostra
    c2 = np.sqrt(ax**2 + az**2)
    c1 = np.sqrt(ax**2 + ay**2 + az**2)

    rows = []

    # metadata fixo do trial
    trial_id = g["trial_id"].iloc[0]
    subject_id = g["subject_id"].iloc[0]
    age_group = g["age_group"].iloc[0]
    activity_code = g["activity_code"].iloc[0]
    rep = int(g["rep"].iloc[0])

    for s, e in sliding_windows_indices(len(g), win_size, step):
        ax_w = ax[s:e]
        ay_w = ay[s:e]
        az_w = az[s:e]

        # stds
        sx = np.std(ax_w, ddof=0)
        sy = np.std(ay_w, ddof=0)
        sz = np.std(az_w, ddof=0)

        # C8, C9
        c8 = np.sqrt(sx**2 + sz**2)
        c9 = np.sqrt(sx**2 + sy**2 + sz**2)

        # C2_max e C1_max
        c2_max = float(np.max(c2[s:e]))
        c1_max = float(np.max(c1[s:e]))

        # label da janela (majority)
        y_win = int(np.round(np.mean(y[s:e])))

        # tempo da janela (início)
        time_start = float(g["time_s"].iloc[s]) if "time_s" in g.columns else float(s / fs)

        rows.append({
            "trial_id": trial_id,
            "subject_id": subject_id,
            "age_group": age_group,
            "activity_code": activity_code,
            "rep": rep,
            "t_start": time_start,
            "c2_max": c2_max,
            "c8": float(c8),
            "c9": float(c9),
            "c1_max": c1_max,
            "label": y_win
        })

    return pd.DataFrame(rows)

def extract_window_features_one_df(df, fs=200.0, win_sec=1.0, step_sec=0.1):
    win = int(fs*win_sec)
    step = int(fs*step_sec)

    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    c1, c2 = compute_c_features_from_axes(ax, ay, az)

    c2_max_list, c8_list, c9_list, c1_max_list = [], [], [], []
    for s,e in sliding_windows_indices(len(df), win, step):
        ax_w, ay_w, az_w = ax[s:e], ay[s:e], az[s:e]
        sx, sy, sz = np.std(ax_w), np.std(ay_w), np.std(az_w)

        c8 = np.sqrt(sx**2 + sz**2)
        c9 = np.sqrt(sx**2 + sy**2 + sz**2)

        c2_max_list.append(np.max(c2[s:e]))
        c1_max_list.append(np.max(c1[s:e]))
        c8_list.append(c8)
        c9_list.append(c9)

    return pd.DataFrame({
        "c2_max": c2_max_list,
        "c8": c8_list,
        "c9": c9_list,
        "c1_max": c1_max_list,
    })

from scipy.stats import skew, kurtosis

def extract_comprehensive_features_one_df_v2(df, fs=200.0, win_sec=3.0, step_sec=1.5):
    # Aumentei a janela para 3.0s para capturar um ciclo completo de passos
    win = int(fs * win_sec)
    step = int(fs * step_sec)
    
    # Lista de eixos disponíveis
    axes = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    # Preparar listas para armazenar
    features_list = []
    
    for s, e in sliding_windows_indices(len(df), win, step):
        window = df.iloc[s:e]
        
        # Dicionário para esta janela
        row_feats = {}
        
        # 1. Features Estatísticas por Eixo (Captura o "Jeito" de andar)
        for col in axes:
            if col in window.columns:
                arr = window[col].values
                row_feats[f'{col}_mean'] = np.mean(arr) # Orientação
                row_feats[f'{col}_std'] = np.std(arr)   # Intensidade/Oscilação
                row_feats[f'{col}_max'] = np.max(arr)
                row_feats[f'{col}_min'] = np.min(arr)
                # Skewness e Kurtosis ajudam muito na identificação
                row_feats[f'{col}_skew'] = skew(arr) 
                row_feats[f'{col}_kurt'] = kurtosis(arr)
        
        # 2. Features Físicas (As que você já tinha, mantemos pois são úteis)
        # Recalcular C2, etc. aqui dentro se necessário, ou usar as colunas se já existirem
        # Exemplo simples de Magnitude Total
        acc_mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
        row_feats['mag_mean'] = np.mean(acc_mag)
        row_feats['mag_std'] = np.std(acc_mag)
        
        # Metadados necessários para o Join depois
        if 'subject_id' in df.columns:
            row_feats['subject_id'] = df['subject_id'].iloc[0]
            row_feats['trial_id'] = df['trial_id'].iloc[0]
            row_feats['activity_code'] = df['activity_code'].iloc[0]
            
        features_list.append(row_feats)
        
    return pd.DataFrame(features_list)

def extract_hybrid_features_one_df(df, fs=200.0, win_sec=3.0, step_sec=1.5):
    """
    V3: Inclui Estatísticas (Tempo), Magnitudes (Invariante a Rotação) 
    e Frequência (FFT - Ritmo).
    """
    win = int(fs * win_sec)
    step = int(fs * step_sec)
    
    features_list = []
    
    # Pré-calcular Magnitudes (Vetor Resultante) - Crucial para invariância de rotação
    # Se o sensor girar, ax muda, mas acc_mag continua igual.
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    
    # Lista de sinais para extrair
    signals = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']
    
    for s, e in sliding_windows_indices(len(df), win, step):
        window = df.iloc[s:e]
        
        if len(window) < win: continue
        
        # Filtro de Silêncio: Se a variância da magnitude for muito baixa, a pessoa está parada.
        if window['acc_mag'].std() < 0.02: 
            continue

        row = {}
        
        # 1. Domínio do Tempo (Estatísticas)
        for col in signals:
            arr = window[col].values
            row[f'{col}_mean'] = np.mean(arr)
            row[f'{col}_std']  = np.std(arr)
            row[f'{col}_max']  = np.max(arr)
            row[f'{col}_min']  = np.min(arr)
            row[f'{col}_iqr']  = iqr(arr) # Intervalo Interquartil (Robusto a outliers)
            row[f'{col}_skew'] = skew(arr)
            row[f'{col}_kurt'] = kurtosis(arr)
            # Zero Crossing Rate (bom para frequência estimada no tempo)
            row[f'{col}_zcr'] = ((arr[:-1] * arr[1:]) < 0).sum()

        # 2. Domínio da Frequência (FFT) - O "Ritmo" da pessoa
        # Apenas na Magnitude para economizar computação e ser robusto a rotação
        for col in ['acc_mag', 'gyro_mag']:
            arr = window[col].values
            # Remover componente DC (gravidade média) para focar na oscilação
            arr_no_dc = arr - np.mean(arr)
            
            # FFT
            fft_vals = np.abs(np.fft.rfft(arr_no_dc))
            fft_freqs = np.fft.rfftfreq(len(arr_no_dc), d=1/fs)
            
            # Dominant Frequency (Cadência da passada)
            dom_idx = np.argmax(fft_vals)
            row[f'{col}_dom_freq'] = fft_freqs[dom_idx]
            
            # Energy (Força da passada)
            row[f'{col}_spec_energy'] = np.sum(fft_vals**2) / len(fft_vals)
            
            # Spectral Entropy (Complexidade do movimento)
            psd_norm = fft_vals / np.sum(fft_vals)
            # Evitar log(0)
            psd_norm = psd_norm[psd_norm > 0]
            row[f'{col}_spec_entropy'] = -np.sum(psd_norm * np.log(psd_norm))

        # Metadados
        if 'subject_id' in df.columns:
            row['subject_id'] = df['subject_id'].iloc[0]
            row['trial_id'] = df['trial_id'].iloc[0]
            row['activity_code'] = df['activity_code'].iloc[0]
            
        features_list.append(row)
        
    return pd.DataFrame(features_list)