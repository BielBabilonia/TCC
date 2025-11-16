import argparse
import csv
import os
import time
import serial
from serial.tools import list_ports
from datetime import datetime

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
    # Caminho padrão da pasta
    pasta_padrao = os.path.join("TCC", "data", "raw")

    ap = argparse.ArgumentParser(description="Coletor de CSV do Pico via Serial")
    ap.add_argument("--porta", default="COM4", help="Porta serial (ex.: COM4 no Windows, /dev/ttyACM0 no Linux)")
    ap.add_argument("--baud", type=int, default=115200, help="Baudrate")
    ap.add_argument("--pasta", default=pasta_padrao, help=f"Pasta EXISTENTE onde salvar os arquivos CSV (padrão: {pasta_padrao})")
    ap.add_argument("--prefixo", default="temperatura", help="Prefixo do nome do arquivo CSV")
    args = ap.parse_args()

    # Verifica se a pasta existe
    if not os.path.isdir(args.pasta):
        print(f"❌ Erro: a pasta '{args.pasta}' não existe.")
        print("Crie a pasta antes de rodar o script.")
        return

    # Gera nome do arquivo com a data
    data_str = datetime.now().strftime("%Y-%m-%d")
    nome_arquivo = f"{args.prefixo}_{data_str}.csv"
    caminho_csv = os.path.join(args.pasta, nome_arquivo)

    # Detecta porta
    porta = args.porta
    try:
        ser = serial.Serial(porta, args.baud, timeout=1)
    except Exception:
        print(f"# Falha ao abrir {porta}. Tentando autodetectar…")
        porta = detectar_porta(None)
        ser = serial.Serial(porta, args.baud, timeout=1)

    print(f"# Coletando de {porta} e salvando em: {caminho_csv}")
    print("# Pressione Ctrl+C para encerrar.")
    garantir_cabecalho(caminho_csv)

    with ser, open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while True:
            try:
                linha = ser.readline().decode(errors="ignore").strip()
                if not linha:
                    continue
                # ignora comentários e cabeçalhos vindos do Pico
                if linha.startswith("#") or linha.lower().startswith("timestamp_iso"):
                    print(linha)
                    continue

                partes = [p.strip() for p in linha.split(",")]
                if len(partes) >= 2:
                    writer.writerow(partes[:2])
                    f.flush()
                    print("CSV:", partes[:2])
                else:
                    print("RAW:", linha)
            except KeyboardInterrupt:
                print("\n# Encerrado pelo usuário.")
                break
            except Exception as e:
                print("# erro:", e)
                time.sleep(0.5)

if __name__ == "__main__":
    main()