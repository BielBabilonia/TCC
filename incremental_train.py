import os
import shutil
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from src.io_utils import load_csv_folder
from src.preprocess import filter_signals
from src.features import build_feature_table
from src.labeling import heuristic_labels
from src.eval_utils import report_cls

# Quantidade de novas árvores por incremento
NUM_BOOST_INCREMENT = 50

# Caminhos importantes
CFG_PATH = "configs/base.yaml"
MODEL_PATH = "artifacts/model_incre.json"
NEW_DATA_PATH = "data/new"
PROCESSED_PATH = "data/processed"
ARTIFACTS_PATH = "artifacts"


def load_or_create_model(params):
    """Carrega um booster existente ou cria um novo modelo vazio."""
    booster = xgb.Booster()
    if os.path.exists(MODEL_PATH):
        print(f"Carregando modelo incremental existente: {MODEL_PATH}")
        booster.load_model(MODEL_PATH)
        return booster

    print("Nenhum modelo incremental encontrado. Criando um novo...")
    # modelo inicial vazio (será treinado com os primeiros dados)
    return None


def incremental_train(df_feat, params, booster):
    """Treinamento incremental com booster do XGBoost."""
    feat_cols = [
        c for c in df_feat.columns
        if c not in ["t_start", "t_end", "label",
                     "session_id", "driver_id", "source_file"]
    ]

    X = df_feat[feat_cols].fillna(0.0).values
    y = df_feat["label"].values

    y = df_feat["label"].values

    weights = np.ones_like(y, dtype=float)
    # ajuste o peso da classe agressiva
    weights[y == 2] = 3.0

    dtrain = xgb.DMatrix(X, label=y, weight=weights, feature_names=feat_cols)


    # Se não existir modelo, treina inicial
    if booster is None:
        print("\nTreinando modelo inicial...")
        booster = xgb.train(params, dtrain, num_boost_round=200)
        return booster

    print(f"\nTreinando incrementalmente (+{NUM_BOOST_INCREMENT} árvores)...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_INCREMENT,
        xgb_model=booster
    )
    return booster


def evaluate_increment(df_feat, booster):
    """Avalia como os novos dados são classificados pelo modelo incrementado."""
    feat_cols = [
        c for c in df_feat.columns
        if c not in ["t_start", "t_end", "label",
                     "session_id", "driver_id", "source_file"]
    ]
    X = df_feat[feat_cols].fillna(0.0).values
    y_true = df_feat["label"].values

    dtest = xgb.DMatrix(X, feature_names=feat_cols)
    y_pred = np.argmax(booster.predict(dtest), axis=1)

    rep, cm = report_cls(y_true, y_pred)
    print("\n=== Avaliação Incremental ===")
    print(rep)
    print("Matriz de Confusão Incremental:\n", cm)

    class_names = ["Conservador", "Moderado", "Agressivo"]

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=True
    )
    ax.set_xlabel("Predição da IA")
    ax.set_ylabel("Rótulo Real")
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticklabels(class_names, rotation=0)

    plt.title("Matriz de Confusão – Incremento de Treinamento")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACTS_PATH}/matriz_confusao_incremental.png", dpi=300)
    plt.close()


def main():
    # 1) carregar config
    cfg = yaml.safe_load(Path(CFG_PATH).read_text())

    # 2) detectar novos CSVs
    new_files = list(Path(NEW_DATA_PATH).glob("*.csv"))
    if not new_files:
        print("Nenhum novo CSV encontrado em data/new/. Nada para treinar.")
        return

    print(f"Encontrados {len(new_files)} novos arquivos:")

    for f in new_files:
        print(" -", f)

    # 3) carregar CSVs novos
    df = load_csv_folder(NEW_DATA_PATH)
    print("\nCarregadas novas amostras:", len(df))

    # 4) pipeline padrão
    df_f = filter_signals(df, alpha_T=0.1, ma_I=9)
    df_feat = build_feature_table(df_f, cfg)

    # 5) aplicar heurística de rótulo se necessário
    # NÃO sobrepor labels reais!
    if "label" not in df_feat.columns and "label_classe" in df.columns:
        print("Usando labels reais do CSV (label_classe).")
        df_feat["label"] = df["label_classe"].map({"conservador":0,"moderado":1,"agressivo":2})


    print("Distribuição dos novos dados:")
    print(df_feat["label"].value_counts())

    # 6) parâmetros do XGBoost
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42
    }

    # 7) carregar modelo existente OU criar novo
    booster = load_or_create_model(params)

    # 8) treinamento incremental
    # dentro do main(), depois de montar df_feat
    df_aggr = df_feat[df_feat["label"] == 2]

    # por exemplo, duplicar a classe agressiva
    df_feat_bal = pd.concat([df_feat, df_aggr, df_aggr], ignore_index=True)

    print("Distribuição após oversampling:")
    print(df_feat_bal["label"].value_counts())

    booster = incremental_train(df_feat_bal, params, booster)

    # 9) avaliação incremental
    evaluate_increment(df_feat, booster)

    # 10) salvar novo modelo
    booster.save_model(MODEL_PATH)
    print(f"\nModelo incremental salvo em: {MODEL_PATH}")

    # 11) mover CSVs usados para processed/
    Path(PROCESSED_PATH).mkdir(exist_ok=True)

    for f in new_files:
        shutil.move(str(f), f"{PROCESSED_PATH}/{f.name}")

    print("\nArquivos movidos para data/processed/.")
    print("Treinamento incremental finalizado com sucesso!\n")


if __name__ == "__main__":
    main()
