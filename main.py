# ============================================================
# LEITURA DE TEMPERATURA COM SENSOR DS18B20 NO RASPBERRY PI PICO 2W
# ============================================================
# Este código lê a temperatura do sensor DS18B20 via protocolo OneWire,
# armazena apenas leituras diferentes da anterior (ignora valores repetidos),
# calcula a média de 10 leituras distintas e exibe o valor arredondado
# para uma casa decimal em graus Celsius e Fahrenheit.
# ============================================================

import time
import board
import adafruit_onewire.bus
import adafruit_ds18x20

# --- CONFIGURAÇÃO DO PINO ---
# Conecte o fio DATA do DS18B20 ao pino abaixo (exemplo: GP5)
# Importante: o sensor precisa de um resistor de 4.7kΩ entre DATA e 3.3V
one_wire_pin = board.GP5

# Cria o barramento OneWire usando o pino configurado
ow_bus = adafruit_onewire.bus.OneWireBus(one_wire_pin)

# Faz uma varredura (scan) no barramento para encontrar dispositivos conectados
devices = ow_bus.scan()

# Caso nenhum sensor seja encontrado, o programa avisa e para a execução
if not devices:
    print("Nenhum sensor DS18B20 encontrado! Verifique conexões e resistor de pull-up (4.7kΩ).")
    while True:
        pass

# Pega o primeiro sensor encontrado
ds18 = adafruit_ds18x20.DS18X20(ow_bus, devices[0])

print("Sensor encontrado! Iniciando leituras de temperatura...\n")

# ============================================================
# LOOP PRINCIPAL DE LEITURA
# ============================================================

leituras = []         # Lista que armazenará as últimas 10 leituras distintas
ultima_leitura = None # Armazena o último valor lido para comparação

while True:
    try:
        # Lê a temperatura em graus Celsius do sensor
        temp_c = ds18.temperature
        
        # Arredonda para 1 casa decimal (para comparação justa)
        temp_c = round(temp_c, 1)

        # --- FILTRO: IGNORA VALORES REPETIDOS ---
        # Só adiciona se o valor for diferente da última leitura armazenada
        if temp_c != ultima_leitura:
            leituras.append(temp_c)
            ultima_leitura = temp_c  # Atualiza o valor anterior

            # Mantém apenas as 10 leituras mais recentes
            if len(leituras) > 10:
                leituras.pop(0)

            # Calcula a média quando houver 10 leituras distintas
            if len(leituras) == 10:
                media_c = round(sum(leituras) / len(leituras), 1)
                media_f = round((media_c * 9 / 5) + 32, 1)
                print(f"Média (10 leituras distintas): {media_c} °C | {media_f} °F")
            else:
                print(f"Nova leitura registrada: {temp_c} °C ({len(leituras)}/10)")
        else:
            # Se o valor for igual ao anterior, ignora e apenas informa
            print(f"Ignorado valor repetido: {temp_c} °C")

    except Exception as e:
        print(f"Erro na leitura: {e}")

    # Aguarda 3 segundos antes da próxima leitura
    time.sleep(3)