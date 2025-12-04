import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["timestamp","I_A","T_pneu"]

def load_csv_folder(folder):
    folder = Path(folder)
    dfs = []
    for f in folder.glob("*.csv"):
        df = pd.read_csv(f)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{f} faltando colunas: {missing}")
        # opcional: V_V, SOC_pct, session_id, driver_id
        df["source_file"] = f.name
        dfs.append(df)
    if not dfs:
        raise ValueError("Nenhum CSV encontrado em data/raw/")
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
