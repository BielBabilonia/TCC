import argparse
import csv
import os
import time
import serial
from serial.tools import list_ports

def detectar_porta(preferida=None):
    portas = [p.device for p in list_ports.comports()]
    if preferida and preferida in portas:
        return preferida
    if portas:
        print(f"# Portas disponíveis: {portas}")
        return portas[0]
    raise RuntimeError("Nenhuma porta serial encontrada.")

def garantir_cabecalho(path, header=("timestamp_iso","temperatura_c")):
    precisa = True
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            primeira = f.readline().strip()
            if primeira.replace(";", ",").lower().startswith(",".join(header)):
                precisa = False
    if precisa:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def main():
    ap = argparse.ArgumentParser(description="Coletor de CSV do Pico via Serial")
    ap.add_argument("--porta", default="COM4", help="Porta serial (ex.: COM4 no Windows, /dev/ttyACM0 no Linux)")
    ap.add_argument("--baud", type=int, default=115200, help="Baudrate")
    ap.add_argument("--saida", default="dados_temperatura.csv", help="Arquivo CSV de saída")
    args = ap.parse_args()

    # Detecta porta se a padrão falhar
    porta = args.porta
    try:
        ser = serial.Serial(porta, args.baud, timeout=1)
    except Exception:
        print(f"# Falha ao abrir {porta}. Tentando autodetectar…")
        porta = detectar_porta(None)
        ser = serial.Serial(porta, args.baud, timeout=1)

    print(f"# Coletando de {porta} em {args.saida}. Ctrl+C para parar.")
    garantir_cabecalho(args.saida)

    with ser, open(args.saida, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while True:
            try:
                linha = ser.readline().decode(errors="ignore").strip()
                if not linha:
                    continue
                # ignora comentários/diagnósticos e o cabeçalho que vem do Pico
                if linha.startswith("#") or linha.lower().startswith("timestamp_iso"):
                    print(linha)
                    continue

                # esperamos "timestamp_iso,temperatura_c"
                partes = [p.strip() for p in linha.split(",")]
                if len(partes) >= 2:
                    writer.writerow(partes[:2])
                    f.flush()
                    print("CSV:", partes[:2])
                else:
                    # loga o que veio diferente
                    print("RAW:", linha)
            except KeyboardInterrupt:
                print("\n# Encerrado pelo usuário.")
                break
            except Exception as e:
                print("# erro:", e)
                time.sleep(0.5)

if __name__ == "__main__":
    main()