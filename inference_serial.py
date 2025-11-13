import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import serial

# módulos do projeto
from src.preprocess import filter_signals
from src.features import extract_features_window

LABEL_MAP = {0: "conservador", 1: "moderado", 2: "agressivo"}

def load_artifact(path):
    art = joblib.load(path)
    return art["model"], art["features"], art.get("cfg", {})

def ensure_cfg(user_cfg_path, train_cfg):
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
    dfw = filter_signals(df_window, alpha_T=0.1, ma_I=9)
    feats = extract_features_window(dfw, cfg)
    return feats

def classify(feats_row, model, feature_names, want_prob=False):
    X = np.array([[feats_row.get(f, 0.0) for f in feature_names]], dtype=float)
    if want_prob and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        yhat = int(np.argmax(proba))
        return yhat, proba
    yhat = int(model.predict(X)[0])
    return yhat, None

def open_serial(port, baud):
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
        return ser
    except Exception as e:
        raise SystemExit(f"Não foi possível abrir a serial '{port}' @ {baud}: {e}")

def parse_line(line, headers=None):
    """
    Aceita:
      - JSON: {"timestamp":..., "I_A":..., "T_pneu":...}
      - CSV:  timestamp,I_A,T_pneu   (se headers informados)
      - CSV sem timestamp: I_A,T_pneu (timestamp gerado localmente)
    """
    line = line.strip()
    if not line:
        return None

    # tenta JSON
    if line.startswith("{") and line.endswith("}"):
        try:
            rec = json.loads(line)
            return rec
        except Exception:
            return None

    # CSV
    parts = [p.strip() for p in line.split(",")]
    if headers:
        if len(parts) != len(headers):
            return None
        rec = {}
        for h, v in zip(headers, parts):
            if h in ("timestamp", "I_A", "T_pneu", "V_V", "SOC_pct"):
                try:
                    rec[h] = float(v)
                except Exception:
                    rec[h] = np.nan
            else:
                rec[h] = v
        return rec
    else:
        # se não há headers, tenta supor 2 ou 3 colunas
        # Formato 1: I_A, T_pneu
        # Formato 2: timestamp, I_A, T_pneu
        if len(parts) == 2:
            try:
                I_A = float(parts[0]); T_pneu = float(parts[1])
                return {"I_A": I_A, "T_pneu": T_pneu}
            except Exception:
                return None
        elif len(parts) == 3:
            try:
                ts = float(parts[0]); I_A = float(parts[1]); T_pneu = float(parts[2])
                return {"timestamp": ts, "I_A": I_A, "T_pneu": T_pneu}
            except Exception:
                return None
    return None

def main():
    ap = argparse.ArgumentParser(description="Inferência em tempo real via porta serial (I_A + T_pneu)")
    ap.add_argument("--port", default="COM4", help="porta serial padrão (pode ser sobrescrita)")
    ap.add_argument("--baud", type=int, default=115200, help="baud rate (padrão: 115200)")
    ap.add_argument("--artifact", default="artifacts/model_joblib.pkl", help="modelo salvo")
    ap.add_argument("--cfg", default="configs/base.yaml", help="YAML com janelas/limiares (opcional)")
    ap.add_argument("--interval", type=float, default=1.0, help="período de saída (s)")
    ap.add_argument("--headers", help="linha de cabeçalho CSV (ex.: 'timestamp,I_A,T_pneu')")
    ap.add_argument("--print-prob", action="store_true", help="exibir probabilidades por classe")
    args = ap.parse_args()

    model, feat_names, train_cfg = load_artifact(args.artifact)
    cfg = ensure_cfg(args.cfg, train_cfg)
    window_sec = cfg.get("window_sec", 5.0)

    ser = open_serial(args.port, args.baud)
    if args.headers:
        headers = [h.strip() for h in args.headers.split(",")]
    else:
        headers = None

    # buffer de amostras
    df_buf = pd.DataFrame(columns=["timestamp", "I_A", "T_pneu", "V_V", "SOC_pct"])
    t0_wall = time.monotonic()

    last_emit = time.monotonic()

    print(f"Inferência ativa na serial '{args.port}' @ {args.baud}. Pressione Ctrl+C para parar.")
    try:
        while True:
            raw = ser.readline().decode(errors="ignore")
            if not raw:
                # sem dados neste ciclo; checa se é hora de emitir
                now = time.monotonic()
                if (now - last_emit) >= max(args.interval, 0.05) and not df_buf.empty:
                    win = last_window(df_buf, window_sec)
                    feats = compute_features_for_window(win, cfg)
                    if feats:
                        yhat, proba = classify(feats, model, feat_names, want_prob=args.print_prob)
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
                continue

            rec = parse_line(raw, headers=headers)
            if rec is None:
                continue

            # gerar timestamp local se ausente (segundos desde início)
            if "timestamp" not in rec or pd.isna(rec["timestamp"]):
                rec["timestamp"] = time.monotonic() - t0_wall

            # garantir colunas
            for k in ["I_A", "T_pneu", "V_V", "SOC_pct"]:
                if k not in rec:
                    rec[k] = np.nan

            # append
            df_buf = pd.concat([df_buf, pd.DataFrame([rec])], ignore_index=True)
            df_buf = df_buf.sort_values("timestamp").reset_index(drop=True)

            # emissão periódica controlada por --interval (já tratada acima)

    except KeyboardInterrupt:
        print("\nParando inferência…")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
