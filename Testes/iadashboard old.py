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

# XGBoost - Booster incremental
import xgboost as xgb

# Pipeline do seu projeto
from src.preprocess import filter_signals
from src.features import extract_features_window
from src.features import build_feature_table
from src.io_utils import load_csv_folder

LABEL_MAP = {0: "conservador", 1: "moderado", 2: "agressivo"}


# ============================================================
# FLASK + SOCKETIO
# ============================================================
app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/')
def index():
    return render_template('dashboard2teste.html')


# ============================================================
# FERRAMENTAS DO MODELO INCREMENTAL
# ============================================================
def load_incremental_model(model_path="artifacts/model_incre.json"):
    """Carrega booster incremental salvo."""
    booster = xgb.Booster()
    booster.load_model(model_path)
    print(f"[OK] Modelo incremental carregado: {model_path}")
    return booster


def classify_booster(feats_row, booster, feature_names):
    """Classificação usando booster incremental (XGBoost)."""
    X = np.array([[feats_row.get(f, 0.0) for f in feature_names]], dtype=float)
    dtest = xgb.DMatrix(X, feature_names=feature_names)
    pred = booster.predict(dtest)[0]
    yhat = int(np.argmax(pred))
    return yhat, pred


def safe_number(x, lim_min=None, lim_max=None):
    """Garante que x é número válido antes de enviar ao dashboard."""
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    if lim_min is not None and x < lim_min:
        return None
    if lim_max is not None and x > lim_max:
        return None
    return x


# ============================================================
# LEITURA DO CSV
# ============================================================
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
    """Extrai features de uma janela de tempo."""
    if df_window.empty or len(df_window) < 2:
        return None
    dfw = filter_signals(df_window, alpha_T=0.1, ma_I=9)
    feats = extract_features_window(dfw, cfg)
    return feats


# ============================================================
# THREAD DE INFERÊNCIA IA
# ============================================================
def ia_inference_thread(args):

    # Carregar booster incremental
    booster = load_incremental_model("artifacts/model_incre.json")

    # Carregar features e cfg do artifact joblib (apenas metadados)
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

            # -------------------------------------
            # JANELA DESLIZANTE
            # -------------------------------------
            win = last_window(df_buf, window_sec)

            # -------------------------------------
            # FEATURES
            # -------------------------------------
            feats = compute_features_for_window(win, cfg)
            if feats is None:
                time.sleep(5)
                continue

            # -------------------------------------
            # CLASSIFICAÇÃO
            # -------------------------------------
            yhat, proba = classify_booster(feats, booster, feat_names)

            # -------------------------------------
            # PEGAR VALORES CRUS DO CSV
            # -------------------------------------
            temp_raw = win["T_pneu"].iloc[-1] if "T_pneu" in win.columns else None
            current_raw = win["I_A"].iloc[-1] if "I_A" in win.columns else None

            temp = safe_number(temp_raw, lim_min=0, lim_max=200)
            current = safe_number(current_raw, lim_min=0, lim_max=400)

            if temp is None or current is None:
                time.sleep(5)
                continue

            # -------------------------------------
            # ALERTAS
            # -------------------------------------
            alerta = ""
            if temp > 80 and current > 200:
                alerta = "Condução extremamente agressiva!"
            elif temp > 80:
                alerta = "Superaquecimento do pneu!"
            elif current > 200:
                alerta = "Corrente muito alta (aceleração agressiva)."

            if yhat == 2:
                alerta = "Piloto classificado como AGRESSIVO pela IA!"

            # -------------------------------------
            # ENVIAR PARA O DASHBOARD
            # -------------------------------------
            out = {
                "temp": float(temp),
                "current": float(current),
                "classe": LABEL_MAP.get(yhat, "desconhecido"),
                "alerta": alerta
            }

            socketio.emit('atualiza_dados', out)

            # Delay fixo entre leituras
            time.sleep(5)


# ============================================================
# MODO ALEATÓRIO (para testes)
# ============================================================
def enviar_dados_randomicos():
    while True:
        temp = random.randint(40, 120)
        current = random.randint(70, 400)
        classe = "conservador"

        if temp > 80 or current > 200:
            classe = "agressivo"

        socketio.emit("atualiza_dados", {
            "temp": temp,
            "current": current,
            "classe": classe,
            "alerta": ""
        })
        time.sleep(2)


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true',
                        help='Simular dados aleatórios')
    parser.add_argument('--artifact', type=str,
                        default="artifacts/model_joblib.pkl")
    parser.add_argument('--csv', type=str,
                        default="data/raw/tcc_sintetico_agressivo.csv")
    args = parser.parse_args()

    if args.random:
        t = threading.Thread(target=enviar_dados_randomicos)
    else:
        t = threading.Thread(target=ia_inference_thread, args=(args,))

    t.daemon = True
    t.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
