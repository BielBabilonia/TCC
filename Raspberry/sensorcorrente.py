import time
import board
import analogio

# --- CONFIGURAÇÃO DO PINO ANALÓGICO ---
# Ligue o pino OUT do ACS712 em GP26 (ADC0)
acs = analogio.AnalogIn(board.A0)

# --- CONFIG. DO MODELO DO SENSOR ---
SENSIBILIDADE = 0.066   # V/A (para ACS712 20A)
# Modelos:
# 5A  → 0.185
# 20A → 0.100
# 30A → 0.066

# --- FUNÇÕES ÚTEIS ---
def ler_tensao(pino):
    # Converte o ADC (0–65535) para tensão (0–3.3V)
    return (pino.value * 3.3) / 65535

def corrente_acs712():
    Vout = ler_tensao(acs)

    # Offset do ACS712 é ~Vcc/2 (por padrão 1.65V)
    Vzero = 3.3 / 2

    # Cálculo da corrente (A)
    I = (Vout - Vzero) / SENSIBILIDADE
    return I, Vout

print("Lendo ACS712...")

while True:
    corrente, tensao = corrente_acs712()
    print(f"Tensão: {tensao:.3f} V  |  Corrente: {corrente:.3f} A")
    time.sleep(0.2)
