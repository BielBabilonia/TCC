from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import random
import argparse

import numpy as np
import pandas as pd
import yaml
import math
import joblib
from pathlib import Path

import xgboost as xgb

from src.preprocess import filter_signals
from src.features import extract_features_window

LABEL_MAP = {0: "conservador", 1: "moderado", 2: "agressivo"}

# ---------------------------------------------------------
# FLASK + SOCKETIO
# ---------------------------------------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/')
def index():
    return render_template('dashboard2teste.html')


# ---------------------------------------------------------
# FUNÇÕES DE MODELO / UTILIDADES
# ---------------------------------------------------------
def load_incremental_model(model_path="artifacts/model_incre.json"):
    booster = xgb.Booster()
    booster.load_model(model_path)
    print(f"[OK] Modelo incremental carregado: {model_path}")
    return booster


def classify_booster(feats_row, booster, feature_names):
    X = np.array([[feats_row.get(f, 0.0) for f in feature_names]], dtype=float)
    dtest = xgb.DMatrix(X, feature_names=feature_names)
    pred = booster.predict(dtest)[0]
    yhat = int(np.argmax(pred))
    return yhat, pred


def safe_number(x, lim_min=None, lim_max=None):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    if lim_min is not None and x < lim_min:
        return None
    if lim_max is not None and x > lim_max:
        return None
    return x


# ---------------------------------------------------------
# LEITURA DO CSV
# ---------------------------------------------------------
def stream_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV precisa ter coluna 'timestamp'")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for _, row in df.iterrows():
        yield row.to_dict()


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


# ---------------------------------------------------------
# THREAD DE INFERÊNCIA IA
# ---------------------------------------------------------
def ia_inference_thread(args):
    # booster incremental
    booster = load_incremental_model("artifacts/model_incre.json")

    # carregar features + cfg do artifact
    artifact = joblib.load(args.artifact)
    feat_names = artifact["features"]
    cfg = artifact["cfg"]
    window_sec = cfg.get("window_sec", 5.0)

    print("[OK] Features carregadas:", len(feat_names))

    df_buf = pd.DataFrame(columns=[
        "timestamp", "I_A", "T_pneu", "V_V", "SOC_pct",
        "session_id", "driver_id", "source_file"
    ])

    while True:
        print("[INFO] Reiniciando leitura do CSV...")
        source_iter = stream_from_csv(args.csv)

        for rec in source_iter:
            row_df = pd.DataFrame([rec])

            if df_buf.empty:
                df_buf = row_df
            else:
                df_buf = pd.concat([df_buf, row_df], ignore_index=True)

            df_buf = df_buf.sort_values("timestamp").reset_index(drop=True)

            # --- 1) criar a janela ---
            win = last_window(df_buf, window_sec)

            # --- 2) extrair features ---
            feats = compute_features_for_window(win, cfg)
            if feats is None:
                time.sleep(5)
                continue

            # classificação
            yhat, proba = classify_booster(feats, booster, feat_names)

            # valores crus
            temp_raw = win["T_pneu"].iloc[-1] if "T_pneu" in win.columns else None
            current_raw = win["I_A"].iloc[-1] if "I_A" in win.columns else None

            temp = safe_number(temp_raw, lim_min=40.0, lim_max=200.0)
            current = safe_number(current_raw, lim_min=70.0, lim_max=400.0)

            if temp is None or current is None:
                time.sleep(5)
                continue

            # alertas
            alerta = ""
            if temp > 80 and current > 200:
                alerta = "Condução extremamente agressiva!"
            elif temp > 80:
                alerta = "Superaquecimento do pneu!"
            elif current > 200:
                alerta = "Corrente muito alta (aceleração agressiva)."

            if yhat == 2:
                alerta = "Piloto classificado como AGRESSIVO pela IA!"

            out = {
                "temp": float(temp),
                "current": float(current),
                "classe": LABEL_MAP.get(yhat, "desconhecido"),
                "alerta": alerta,
            }

            socketio.emit('atualiza_dados', out)

            # delay fixo de 5 s entre emissões
            time.sleep(5)


# ---------------------------------------------------------
# MODO ALEATÓRIO (TESTE)
# ---------------------------------------------------------
def enviar_dados_randomicos():
    while True:
        temp = random.uniform(40, 120)
        current = random.uniform(70, 400)

        classe = "conservador"
        alerta = ""
        if temp > 80 and current > 200:
            classe = "agressivo"
            alerta = "Condução extremamente agressiva!"
        elif temp > 80:
            alerta = "Superaquecimento do pneu!"
        elif current > 200:
            alerta = "Corrente muito alta (aceleração agressiva)."

        socketio.emit("atualiza_dados", {
            "temp": float(temp),
            "current": float(current),
            "classe": classe,
            "alerta": alerta,
        })
        time.sleep(2)


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true',
                        help='Simular dados aleatórios')
    parser.add_argument('--artifact', type=str,
                        default="artifacts/model_joblib.pkl")
    parser.add_argument('--csv', type=str,
                        default="data/raw/tcc_sintetico_realista.csv")
    args = parser.parse_args()

    if args.random:
        t = threading.Thread(target=enviar_dados_randomicos)
    else:
        t = threading.Thread(target=ia_inference_thread, args=(args,))

    t.daemon = True
    t.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
