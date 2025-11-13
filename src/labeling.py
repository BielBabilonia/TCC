import numpy as np
import pandas as pd

# 0 = conservador, 1 = moderado, 2 = agressivo
def heuristic_labels(df_feat, by="session_id"):
    df = df_feat.copy()
    group_key = by if by in df.columns else None

    def label_group(g):
        # percentis por grupo
        p80_I = np.percentile(g["time_above_thr_I"].fillna(0), 80)
        p95_picoI = np.percentile(g["p99_I"].fillna(0), 95)  # usa p99 como pico proxy
        p90_dIdt = np.percentile(g["dIdt_max"].fillna(0), 90)
        p80_dTdt = np.percentile(g["dTdt_mean"].fillna(0), 80)
        p20_meanI = np.percentile(g["mean_I"].fillna(0), 20)
        p20_varI  = np.percentile(g["std_I"].fillna(0), 20)
        p20_dTdt  = np.percentile(g["dTdt_mean"].fillna(0), 20)
        p20_peaks = np.percentile(g["n_peaks_I"].fillna(0), 20)

        labels = []
        for _, r in g.iterrows():
            cond_aggr = sum([
                r.get("time_above_thr_I",0) > p80_I,
                r.get("p99_I",0) > p95_picoI or r.get("dIdt_max",0) > p90_dIdt,
                r.get("dTdt_mean",0) > p80_dTdt
            ]) >= 2

            cond_cons = all([
                r.get("mean_I",0) < p20_meanI,
                r.get("std_I",0) < p20_varI,
                r.get("dTdt_mean",0) < p20_dTdt,
                r.get("n_peaks_I",0) <= p20_peaks
            ])

            if cond_aggr:
                labels.append(2)
            elif cond_cons:
                labels.append(0)
            else:
                labels.append(1)
        g = g.copy()
        g["label"] = labels
        return g

    if group_key:
        parts = []
        for _, g in df.groupby(group_key):
            parts.append(label_group(g))
        return pd.concat(parts, ignore_index=True)
    else:
        return label_group(df)
