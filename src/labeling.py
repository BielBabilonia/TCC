import numpy as np
import pandas as pd

# Mapa de classes: 0=conservador, 1=moderado, 2=agressivo
CLASS_NAMES = {0: "conserv.", 1: "moder.", 2: "agress."}

def _nanpercentile(a, q):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    return np.percentile(a, q)

def _safe_get(row, key, default=0.0):
    v = row.get(key, default)
    try:
        return float(v)
    except Exception:
        return default

def _label_with_thresholds(g, p):
    """Aplica regras com base nos percentis p (dict) para um DataFrame g."""
    labels = []
    for _, r in g.iterrows():
        time_above = _safe_get(r, "time_above_thr_I")
        p99_I      = _safe_get(r, "p99_I")
        dIdt_max   = _safe_get(r, "dIdt_max")
        dTdt_mean  = _safe_get(r, "dTdt_mean")
        mean_I     = _safe_get(r, "mean_I")
        std_I      = _safe_get(r, "std_I")
        n_peaks    = _safe_get(r, "n_peaks_I")

        cond_aggr = sum([
            time_above > p["p80_time_above"],
            (p99_I > p["p95_p99I"]) or (dIdt_max > p["p90_dIdt"]),
            dTdt_mean > p["p80_dTdt"],
        ]) >= 2

        cond_cons = all([
            mean_I < p["p20_meanI"],
            std_I  < p["p20_stdI"],
            dTdt_mean < p["p20_dTdt"],
            n_peaks <= p["p20_peaks"],
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

def _compute_thresholds(g):
    """Calcula percentis necessários para o grupo g."""
    return {
        "p80_time_above": _nanpercentile(g.get("time_above_thr_I", []), 80),
        "p95_p99I":       _nanpercentile(g.get("p99_I", []), 95),
        "p90_dIdt":       _nanpercentile(g.get("dIdt_max", []), 90),
        "p80_dTdt":       _nanpercentile(g.get("dTdt_mean", []), 80),
        "p20_meanI":      _nanpercentile(g.get("mean_I", []), 20),
        "p20_stdI":       _nanpercentile(g.get("std_I", []), 20),
        "p20_dTdt":       _nanpercentile(g.get("dTdt_mean", []), 20),
        "p20_peaks":      _nanpercentile(g.get("n_peaks_I", []), 20),
    }

def _scale01(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmin(x) if np.any(~np.isnan(x)) else 0.0
    M = np.nanmax(x) if np.any(~np.isnan(x)) else 1.0
    if not np.isfinite(m) or not np.isfinite(M) or M - m <= 1e-12:
        return np.zeros_like(x)
    return (x - m) / (M - m)

def _fallback_quantile_labels(df):
    """
    Se ainda assim o dataset final ficar com ≤1 classe, cria um score de agressividade:
    S = w1*time_above + w2*p99_I + w3*dIdt_max + w4*dTdt_mean (normalizados 0–1),
    e rotula por tercis (0/1/2).
    """
    w1, w2, w3, w4 = 0.35, 0.30, 0.20, 0.15
    s1 = _scale01(df.get("time_above_thr_I", 0))
    s2 = _scale01(df.get("p99_I", 0))
    s3 = _scale01(df.get("dIdt_max", 0))
    s4 = _scale01(df.get("dTdt_mean", 0))
    S = w1*s1 + w2*s2 + w3*s3 + w4*s4

    q1 = _nanpercentile(S, 33.33)
    q2 = _nanpercentile(S, 66.67)

    labels = np.full(len(df), 1, dtype=int)  # default moderado
    labels[S <= q1] = 0
    labels[S >= q2] = 2
    out = df.copy()
    out["label"] = labels
    return out

def heuristic_labels(
    df_feat: pd.DataFrame,
    by: str = "session_id",
    fallback_global: bool = True,
    ensure_diversity: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rótulo heurístico com tolerância:
      1) calcula percentis por grupo (ex.: sessão) e rotula;
      2) se algum grupo ficar mono-classe, recalcula com percentis globais;
      3) se o total ainda tiver ≤1 classe, aplica fallback por quantis de score.
    """
    df = df_feat.copy()

    # ===== 1) Label por grupo =====
    groups = [("", df)] if by not in df.columns else list(df.groupby(by))
    parts = []
    for gname, g in groups:
        p = _compute_thresholds(g)
        g_lab = _label_with_thresholds(g, p)
        if verbose:
            uniq = np.unique(g_lab["label"])
            print(f"[heuristic_labels] Grupo '{gname or 'GLOBAL'}' → classes: {uniq.tolist()}")
        parts.append(g_lab)

    df_out = pd.concat(parts, ignore_index=True)

    # ===== 2) Fallback por grupo com percentis GLOBAIS (se necessário) =====
    if fallback_global and by in df.columns:
        global_p = _compute_thresholds(df_out)
        fixed_parts = []
        for gname, g in df_out.groupby(by):
            if len(np.unique(g["label"])) < 2:
                if verbose:
                    print(f"[heuristic_labels] Fallback global no grupo '{gname}' (mono-classe).")
                g = _label_with_thresholds(g.drop(columns=["label"]), global_p)
            fixed_parts.append(g)
        df_out = pd.concat(fixed_parts, ignore_index=True)

    # ===== 3) Fallback final por score se ainda pobre =====
    if ensure_diversity and len(np.unique(df_out["label"])) < 2:
        if verbose:
            print("[heuristic_labels] Fallback final por score (tercis).")
        df_out = _fallback_quantile_labels(df_out)

    if verbose:
        vc = df_out["label"].value_counts().sort_index()
        print("[heuristic_labels] Distribuição final de classes:")
        for c in [0,1,2]:
            print(f"  {c} ({CLASS_NAMES[c]}): {int(vc.get(c, 0))}")

    return df_out
