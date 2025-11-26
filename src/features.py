import numpy as np
import pandas as pd

def _d_dt(x, t):
    # derivada simples (diferenças sucessivas)
    dx = np.diff(x)
    dt = np.diff(t)
    dt[dt==0] = 1e-6
    return dx/dt

def _count_peaks_above(series, thr):
    # contagem de amostras acima do limiar (proxy de "picos")
    return int((series > thr).sum())

def extract_features_window(chunk, cfg):
    feats = {}
    t = chunk["timestamp"].values

    # Corrente filtrada
    I = chunk.get("I_A_f", chunk["I_A"]).values
    if len(I) >= 2:
        dIdt = _d_dt(I, t)
        feats["mean_I"] = float(np.mean(I))
        feats["std_I"]  = float(np.std(I, ddof=1)) if len(I)>1 else 0.0
        feats["rms_I"]  = float(np.sqrt(np.mean(I**2)))
        feats["p95_I"]  = float(np.percentile(I, 95))
        feats["p99_I"]  = float(np.percentile(I, 99))
        feats["dIdt_max"] = float(np.max(dIdt))
        thr = cfg["i_fraction_threshold"] * cfg["i_nominal_a"]
        feats["n_peaks_I"] = _count_peaks_above(pd.Series(I), thr)
        feats["time_above_thr_I"] = float(np.mean(I > thr))  # fração da janela

    # Temperatura de pneu filtrada
    if "T_pneu_f" in chunk.columns or "T_pneu" in chunk.columns:
        T = chunk.get("T_pneu_f", chunk["T_pneu"]).values
        if len(T) >= 2:
            dTdt = _d_dt(T, t)
            feats["mean_T"] = float(np.mean(T))
            feats["max_T"]  = float(np.max(T))
            feats["deltaT"] = float(np.max(T) - np.min(T))
            feats["dTdt_mean"] = float(np.mean(dTdt))
            feats["dTdt_max"]  = float(np.max(dTdt))

    return feats

def build_feature_table(df, cfg):
    rows = []
    for s, e, chunk in window_iter_hook(df, cfg):
        if len(chunk) < 2:
            continue
        feats = extract_features_window(chunk, cfg)
        feats["t_start"] = s
        feats["t_end"] = e
        # herdar meta se existir
        for meta in ["session_id","driver_id","source_file"]:
            if meta in df.columns:
                feats[meta] = chunk[meta].iloc[0]
        rows.append(feats)
    return pd.DataFrame(rows)

# hook: usa a window_iter do preprocess, mas injeta cfg
from .preprocess import window_iter
def window_iter_hook(df, cfg):
    return window_iter(df, cfg["window_sec"], cfg["hop_sec"])
