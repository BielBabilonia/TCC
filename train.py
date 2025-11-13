import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.io_utils import load_csv_folder
from src.preprocess import filter_signals
from src.features import build_feature_table
from src.labeling import heuristic_labels
from src.eval_utils import report_cls

def main(args):
    # 1) carregar config
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    # 2) ler dados brutos
    df = load_csv_folder("data/raw")
    # 3) filtrar sinais
    df_f = filter_signals(df, alpha_T=0.1, ma_I=9)
    # 4) features por janelas
    df_feat = build_feature_table(df_f, cfg)
    print("Distribuição global (antes da heurística, pode estar vazia):")
    if "label" in df_feat.columns:
        print(df_feat["label"].value_counts(dropna=False))
    # 5) rotulagem heurística (se você ainda não tiver coluna 'label')
    if "label" not in df_feat.columns:
        df_feat = heuristic_labels(df_feat, by="session_id" if "session_id" in df_feat.columns else None)
        print("Distribuição global (antes da heurística, pode estar vazia):")
        if "label" in df_feat.columns:
            print(df_feat["label"].value_counts(dropna=False))

    # 6) montar X/y e grupos (split por sessão/arquivo pra evitar vazamento)
    label_col = "label"
    feat_cols = [c for c in df_feat.columns if c not in ["t_start","t_end","label","session_id","driver_id","source_file"]]
    X = df_feat[feat_cols].fillna(0.0).values
    y = df_feat[label_col].values
    groups = df_feat["session_id"].values if "session_id" in df_feat.columns else df_feat["source_file"].values

    # 7) validação cruzada por grupos
    gkf = GroupKFold(n_splits=5 if len(np.unique(groups))>=5 else max(2, len(np.unique(groups))))
    y_true_all, y_pred_all = [], []
    for tr, te in gkf.split(X, y, groups):
        mdl = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", random_state=42
        )
        mdl.fit(X[tr], y[tr])
        y_pred = mdl.predict(X[te])
        y_true_all.extend(y[te])
        y_pred_all.extend(y_pred)

    rep, cm = report_cls(np.array(y_true_all), np.array(y_pred_all))
    print("\n=== CLASSIFICATION REPORT (macro) ===\n", rep)
    print("=== CONFUSION MATRIX (rows=true, cols=pred) ===\n", cm)

    # 8) treinar final em tudo e salvar
    final_model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        tree_method="hist", random_state=42
    )
    final_model.fit(X, y)
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "features": feat_cols, "cfg": cfg}, "artifacts/model_joblib.pkl")
    print("\nModelo salvo em artifacts/model_joblib.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="pasta de CSVs")
    args = ap.parse_args()
    main(args)
