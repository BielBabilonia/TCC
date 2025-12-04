import numpy as np
import pandas as pd

def ema(series, alpha):
    s = series.astype(float)
    out = np.empty_like(s)
    out[0] = s.iloc[0]
    for i in range(1, len(s)):
        out[i] = alpha * s.iloc[i] + (1 - alpha) * out[i-1]
    return pd.Series(out, index=series.index)

def moving_average(series, k=9):
    return series.rolling(k, min_periods=1, center=True).mean()

def filter_signals(df, alpha_T=0.1, ma_I=9):
    out = df.copy()
    if "I_A" in out.columns:
        out["I_A_f"] = moving_average(out["I_A"], k=ma_I)
    if "T_pneu" in out.columns:
        out["T_pneu_f"] = ema(out["T_pneu"], alpha=alpha_T)
    return out

def window_iter(df, window_sec=5, hop_sec=1):
    # assume timestamp em segundos (float). Se estiver em ms, converta.
    t = df["timestamp"].values
    t0 = t.min()
    t_end = t.max()
    start = t0
    while start + window_sec <= t_end:
        end = start + window_sec
        chunk = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        yield start, end, chunk
        start += hop_sec
