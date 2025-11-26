from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import random
import argparse

# Imports da IA
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from src.preprocess import filter_signals
from src.features import extract_features_window

LABEL_MAP = {0: "conservador", 1: "moderado", 2: "agressivo"}

# --------------------- FLASK ------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('dashboard.html')

# --------------------- RANDOM DATA ------------------
def enviar_dados_randomicos():
    while True:
        temp = random.randint(0, 100)
        current = random.randint(0, 100)

        if temp > 80 and current > 80:
            alerta = "Condução perigosa! Valores muito altos."
            classe = "agressivo"
        elif temp > 80:
            alerta = "Superaquecimento do pneu!"
            classe = "agressivo"
        elif current > 80:
            alerta = "Aceleração muito alta!"
            classe = "agressivo"
        else:
            alerta = ""
            classe = "conservador"

        socketio.emit('atualiza_dados', {
            'temp': temp,
            'current': current,
            'classe': classe,
            'alerta': alerta
        })
        time.sleep(2)

# ------------ IA REAL (USANDO CSV e MODELO) -----------
def load_artifact(artifact_path):
    art = joblib.load(artifact_path)
    mdl = art["model"]
    features = art["features"]
    cfg = art.get("cfg", {})
    return mdl, features, cfg

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

def classify(feats_row, model, feature_names, print_prob=False):
    X = np.array([[feats_row.get(f, 0.0) for f in feature_names]], dtype=float)
    proba = getattr(model, "predict_proba")(X)[0]
    yhat = int(np.argmax(proba))
    if print_prob:
        return yhat, proba
    return yhat, None

def stream_from_csv(csv_path, speed=1.0, time_col="timestamp"):
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"CSV precisa da coluna '{time_col}'")
    df = df.sort_values(time_col).reset_index(drop=True)
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

def ia_inference_thread(args):
    model, feat_names, train_cfg = load_artifact(args.artifact)
    cfg = ensure_cfg(args.cfg, train_cfg)
    window_sec = cfg.get("window_sec", 5.0)
    last_emit = 0.0
    df_buf = pd.DataFrame(columns=["timestamp", "I_A", "T_pneu", "V_V", "SOC_pct",
                                   "session_id", "driver_id", "source_file"])
    source_iter = stream_from_csv(args.csv, speed=args.speed)
    try:
        for rec in source_iter:
            df_buf = pd.concat([df_buf, pd.DataFrame([rec])], ignore_index=True)
            df_buf = df_buf.sort_values("timestamp").reset_index(drop=True)
            now = time.time()
            if (now - last_emit) >= max(args.interval, 0.05):
                win = last_window(df_buf, window_sec)
                feats = compute_features_for_window(win, cfg)
                if feats is None:
                    last_emit = now
                    continue
                yhat, proba = classify(feats, model, feat_names, print_prob=args.print_prob)
                temp = win["T_pneu"].iloc[-1] if "T_pneu" in win else None
                current = win["I_A"].iloc[-1] if "I_A" in win else None
                alerta = ""
                if temp is not None and current is not None:
                    if temp > 80 and current > 80:
                        alerta = "Condução perigosa! Valores muito altos."
                    elif temp > 80:
                        alerta = "Superaquecimento do pneu!"
                    elif current > 80:
                        alerta = "Aceleração muito alta!"
                if yhat == 2:
                    alerta = "Alerta: Direção agressiva por IA!"

                out = {
                    "temp": float(temp) if temp is not None else None,
                    "current": float(current) if current is not None else None,
                    "classe": LABEL_MAP.get(yhat, str(yhat)),
                    "alerta": alerta
                }
                socketio.emit('atualiza_dados', out)
                last_emit = now
    except KeyboardInterrupt:
        print("\nParando inferência…")

# -------- ENTRYPOINT ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', help='Simular dados aleatórios em vez de rodar IA')
    parser.add_argument('--artifact', type=str, default="artifacts/model_joblib.pkl")
    parser.add_argument('--cfg', type=str, default="configs/base.yaml")
    parser.add_argument('--csv', type=str, default="data.csv")
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--print_prob', action='store_true')
    args = parser.parse_args()

    if args.random:
        t = threading.Thread(target=enviar_dados_randomicos)
    else:
        t = threading.Thread(target=ia_inference_thread, args=(args,))
    t.daemon = True
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
