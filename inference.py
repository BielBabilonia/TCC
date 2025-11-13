import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# módulos do projeto
from src.preprocess import filter_signals
from src.features import extract_features_window
# observação: usaremos extract_features_window direto na janela atual

LABEL_MAP = {0: "conservador", 1: "moderado", 2: "agressivo"}

def load_artifact(artifact_path):
    art = joblib.load(artifact_path)
    mdl = art["model"]
    features = art["features"]
    cfg = art.get("cfg", {})
    return mdl, features, cfg

def ensure_cfg(user_cfg_path, train_cfg):
    # dá prioridade ao YAML se fornecido; senão usa cfg salvo no artefato
    if user_cfg_path and Path(user_cfg_path).exists():
        return yaml.safe_load(Path(user_cfg_path).read_text())
    return train_cfg

def last_window(df, window_sec):
    if df.empty:
        return df
    t_end = df["timestamp"].iloc[-1]
    return df[df["timestamp"] >= t_end - window_sec].copy()

def compute_features_for_window(df_window, cfg):
    if df_window.empty or len(df_window) < 2:
        return None
    # filtra sinais na JANELA (leve e suficiente para inferência)
    dfw = filter_signals(df_window, alpha_T=0.1, ma_I=9)
    feats = extract_features_window(dfw, cfg)
    return feats

def classify(feats_row, model, feature_names, print_prob=False):
    # monta vetor X na mesma ordem usada no treino
    X = np.array([[feats_row.get(f, 0.0) for f in feature_names]], dtype=float)
    proba = getattr(model, "predict_proba")(X)[0]
    yhat = int(np.argmax(proba))
    if print_prob:
        return yhat, proba
    return yhat, None

def stream_from_csv(csv_path, speed=1.0, time_col="timestamp"):
    """
    Simula tempo real lendo um CSV ORDENADO por timestamp.
    speed=1.0 -> tempo real; 2.0 -> 2x mais rápido; 0 -> sem dormir.
    """
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"CSV precisa da coluna '{time_col}'")
    df = df.sort_values(time_col).reset_index(drop=True)

    # rebase timestamps para começar em ~agora
    t0 = df[time_col].iloc[0]
    start_wall = time.time()
    for i, row in df.iterrows():
        current_t = row[time_col]
        if speed > 0:
            target_elapsed = (current_t - t0) / max(speed, 1e-9)
            sleep_s = target_elapsed - (time.time() - start_wall)
            if sleep_s > 0:
                time.sleep(sleep_s)
        yield row.to_dict()

def stream_from_stdin():
    """
    Lê linhas de stdin. Aceita CSV simples "timestamp,I_A,T_pneu"
    ou JSON {"timestamp":..,"I_A":..,"T_pneu":..}
    """
    import sys
    headers = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # tenta JSON
        if line.startswith("{") and line.endswith("}"):
            try:
                yield json.loads(line)
                continue
            except Exception:
                pass
        # tenta CSV
        parts = [p.strip() for p in line.split(",")]
        if headers is None:
            headers = parts
            continue
        if headers and len(parts) == len(headers):
            try:
                rec = {h: (float(v) if h in ("timestamp","I_A","T_pneu","V_V","SOC_pct") else v)
                       for h, v in zip(headers, parts)}
                yield rec
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser(description="Real-time inference for driving style (I + T_pneu)")
    ap.add_argument("--artifact", default="artifacts/model_joblib.pkl", help="caminho do modelo salvo")
    ap.add_argument("--cfg", default="configs/base.yaml", help="YAML com janelas/limiares (opcional)")
    ap.add_argument("--csv", help="arquivo CSV para simulação em tempo real")
    ap.add_argument("--stdin", action="store_true", help="ler amostras do stdin (JSON ou CSV streaming)")
    ap.add_argument("--interval", type=float, default=1.0, help="periodicidade de saída (s)")
    ap.add_argument("--speed", type=float, default=1.0, help="fator de velocidade da simulação do CSV (1.0 = tempo real)")
    ap.add_argument("--print-prob", action="store_true", help="exibir probabilidades por classe")
    args = ap.parse_args()

    if not args.csv and not args.stdin:
        raise SystemExit("Escolha uma fonte: --csv <arquivo> OU --stdin")

    model, feat_names, train_cfg = load_artifact(args.artifact)
    cfg = ensure_cfg(args.cfg, train_cfg)

    window_sec = cfg.get("window_sec", 5.0)
    hop_sec = cfg.get("hop_sec", 1.0)  # só para referência; usaremos --interval
    last_emit = 0.0

    buffer = []  # armazenará dicionários
    df_buf = pd.DataFrame(columns=["timestamp","I_A","T_pneu","V_V","SOC_pct","session_id","driver_id","source_file"])

    # escolhe fonte
    if args.csv:
        source_iter = stream_from_csv(args.csv, speed=args.speed)
    else:
        source_iter = stream_from_stdin()

    try:
        for rec in source_iter:
            # garantir colunas básicas
            # campos ausentes viram NaN; manteremos ao construir o df
            df_buf = pd.concat([df_buf, pd.DataFrame([rec])], ignore_index=True)
            df_buf = df_buf.sort_values("timestamp").reset_index(drop=True)

            now = time.time()
            if (now - last_emit) >= max(args.interval, 0.05):
                # pega última janela
                win = last_window(df_buf, window_sec)
                feats = compute_features_for_window(win, cfg)
                if feats is None:
                    last_emit = now
                    continue

                yhat, proba = classify(feats, model, feat_names, print_prob=args.print_prob)
                out = {
                    "t_start": float(win["timestamp"].iloc[0]),
                    "t_end": float(win["timestamp"].iloc[-1]),
                    "classe": LABEL_MAP.get(yhat, str(yhat)),
                }
                if proba is not None:
                    out["prob"] = {
                        "conservador": float(proba[0]) if len(proba) > 0 else None,
                        "moderado":    float(proba[1]) if len(proba) > 1 else None,
                        "agressivo":   float(proba[2]) if len(proba) > 2 else None,
                    }

                print(json.dumps(out, ensure_ascii=False))
                last_emit = now

    except KeyboardInterrupt:
        print("\nParando inferência…")

if __name__ == "__main__":
    main()
