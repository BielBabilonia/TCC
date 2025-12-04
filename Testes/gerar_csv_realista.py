import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ---------------------------------------
# Parâmetros principais
# ---------------------------------------
N_SESSOES_POR_CLASSE = 3      # quantas sessões de cada tipo
N_PONTOS_POR_SESSAO = 200     # duração de cada sessão (segundos)
ARQUIVO_SAIDA = "data/raw/tcc_sintetico_realista.csv"

# se quiser gerar apenas uma classe, use: {"agressivo"}, {"moderado"}, {"conservador"}
CLASSES_ATIVAS = {"agressivo"}
# CLASSES_ATIVAS = {"agressivo"}  # <- descomente esta linha para gerar SÓ agressivo


def gera_trajetoria(classe, sessao_id_base, timestamp_inicio):
    """
    Gera uma sessão com N_PONTOS_POR_SESSAO pontos para uma dada classe,
    com um pouco de tendência e ruído (mais realista).
    """
    linhas = []
    t = timestamp_inicio

    # Parâmetros de faixa média por classe (centro + variação)
    if classe == "conservador":
        I_base = (90, 20)    # média 90A, var 20
        T_base = (47, 6)     # média 47°C, var 6
        V_base = (60, 25)
        SOC_base = (85, 8)
        sessao_nome = "sessao_conservador"

    elif classe == "moderado":
        I_base = (150, 40)   # 150A +/- 40
        T_base = (60, 8)     # 60°C +/- 8
        V_base = (90, 30)
        SOC_base = (70, 15)
        sessao_nome = "sessao_moderado"

    else:  # agressivo
        I_base = (260, 70)   # 260A +/- 70 (pode passar de 400, depois limitamos)
        T_base = (85, 15)    # 85°C +/- 15
        V_base = (120, 35)
        SOC_base = (50, 20)
        sessao_nome = "sessao_agressivo"

    # tendência suave (subida/descida) dentro da sessão
    trend_I = np.linspace(-0.5, 0.5, N_PONTOS_POR_SESSAO)
    trend_T = np.linspace(-0.3, 0.7, N_PONTOS_POR_SESSAO)

    for i in range(N_PONTOS_POR_SESSAO):
        t += 1.0  # 1 segundo

        # ruído gaussiano + tendência
        I_A = np.random.normal(I_base[0], I_base[1]) + trend_I[i] * I_base[0]
        T_pneu = np.random.normal(T_base[0], T_base[1]) + trend_T[i] * 3
        V_V = np.random.normal(V_base[0], V_base[1])
        SOC_pct = np.random.normal(SOC_base[0], SOC_base[1])

        # limites físicos / de projeto
        I_A = np.clip(I_A, 70, 420)        # nunca abaixo de 70A
        T_pneu = np.clip(T_pneu, 40, 130)  # nunca abaixo de 40°C
        V_V = np.clip(V_V, 20, 180)
        SOC_pct = np.clip(SOC_pct, 10, 100)

        driver_id = f"driver_{np.random.randint(1, 4)}"
        source_file = "tcc_sintetico_realista.csv"

        linhas.append({
            "timestamp": t,
            "I_A": I_A,
            "T_pneu": T_pneu,
            "V_V": V_V,
            "SOC_pct": SOC_pct,
            "session_id": f"{sessao_nome}_{sessao_id_base}",
            "driver_id": driver_id,
            "source_file": source_file,
            "label_classe": classe
        })

    return linhas, t


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    todas_linhas = []
    timestamp = 0.0
    sessao_id = 1

    for classe in ["conservador", "moderado", "agressivo"]:
        if classe not in CLASSES_ATIVAS:
            continue

        for _ in range(N_SESSOES_POR_CLASSE):
            linhas, timestamp = gera_trajetoria(classe, sessao_id, timestamp)
            todas_linhas.extend(linhas)
            sessao_id += 1

    df = pd.DataFrame(todas_linhas)

    # Embaralha (mistura classes e sessões)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Arredonda para 3 casas decimais
    cols_float = ["timestamp", "I_A", "T_pneu", "V_V", "SOC_pct"]
    df[cols_float] = df[cols_float].round(3)

    df.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"CSV gerado em: {ARQUIVO_SAIDA}")
    print(df[["timestamp", "I_A", "T_pneu", "label_classe"]].head())


if __name__ == "__main__":
    main()
