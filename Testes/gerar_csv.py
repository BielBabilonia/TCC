import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Quantidade de amostras por classe
n_por_classe = 300   # ajuste se quiser

linhas = []
timestamp = 0.0

for classe in ["conservador", "moderado", "agressivo"]:
    for i in range(n_por_classe):
        timestamp += 1.0   # 1 segundo entre amostras

        if classe == "conservador":
            # Corrente e temperatura dentro do mínimo desejado
            I_A = np.random.uniform(70, 100)       # 70–100 A
            T_pneu = np.random.uniform(40, 50)     # 40–50 ºC
            V_V = np.random.uniform(20, 80)
            SOC_pct = np.random.uniform(60, 100)
            session_id = "sessao_conservador"

        elif classe == "moderado":
            I_A = np.random.uniform(100, 200)      # 100–200 A
            T_pneu = np.random.uniform(50, 70)     # 50–70 ºC
            V_V = np.random.uniform(40, 120)
            SOC_pct = np.random.uniform(40, 80)
            session_id = "sessao_moderado"

        else:  # agressivo
            I_A = np.random.uniform(200, 400)      # 200–400 A
            T_pneu = np.random.uniform(80, 120)    # 80–120 ºC
            V_V = np.random.uniform(80, 160)
            SOC_pct = np.random.uniform(20, 60)
            session_id = "sessao_agressivo"

        driver_id = f"driver_{np.random.randint(1, 4)}"
        source_file = "tcc_sintetico_novo.csv"

        linhas.append({
            "timestamp": timestamp,
            "I_A": I_A,
            "T_pneu": T_pneu,
            "V_V": V_V,
            "SOC_pct": SOC_pct,
            "session_id": session_id,
            "driver_id": driver_id,
            "source_file": source_file,
            "label_classe": classe
        })

# Cria DataFrame
df = pd.DataFrame(linhas)

# Embaralha linhas (mistura as classes no tempo)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

# GARANTIA extra: nunca menos que 70A e 40°C
df["I_A"] = df["I_A"].clip(lower=70.0)
df["T_pneu"] = df["T_pneu"].clip(lower=40.0)

# Arredonda para 3 casas decimais (fica mais limpo)
cols_float = ["timestamp", "I_A", "T_pneu", "V_V", "SOC_pct"]
df[cols_float] = df[cols_float].round(3)

# Garante que a pasta existe
Path("data/raw").mkdir(parents=True, exist_ok=True)

caminho_csv = "data/raw/tcc_sintetico_novo.csv"
df.to_csv(caminho_csv, index=False)

print(f"CSV gerado em: {caminho_csv}")
print(df[["timestamp", "I_A", "T_pneu", "label_classe"]].head())
