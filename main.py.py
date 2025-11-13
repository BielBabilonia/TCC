# ============================================================
# LEITURA DE TEMPERATURA COM SENSOR DS18B20 NO RASPBERRY PI PICO 2W
# ============================================================
# Este código lê a temperatura do sensor DS18B20 via protocolo OneWire,
# calcula a média de 10 leituras consecutivas e exibe o valor arredondado
# para uma casa decimal em graus Celsius e Fahrenheit.
# ============================================================

import time                       # Biblioteca para controlar o tempo e intervalos entre leituras
import board                      # Biblioteca com o mapeamento dos pinos do Raspberry Pi Pico
import adafruit_onewire.bus       # Biblioteca para comunicação OneWire (usada pelo DS18B20)
import adafruit_ds18x20           # Biblioteca específica para o sensor DS18B20

# --- CONFIGURAÇÃO DO PINO DE DADOS ---
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
        pass  # Fica travado aqui até o problema ser corrigido

# Pega o primeiro sensor encontrado no barramento (caso haja mais de um)
ds18 = adafruit_ds18x20.DS18X20(ow_bus, devices[0])

print("Sensor encontrado! Iniciando leituras de temperatura...\n")

# ============================================================
# LOOP PRINCIPAL DE LEITURA
# ============================================================
leituras = []  # Lista que armazenará as últimas 10 leituras de temperatura

while True:
    try:
        # Lê a temperatura em graus Celsius do sensor
        temp_c = ds18.temperature
        
        # Adiciona a leitura atual à lista
        leituras.append(temp_c)
        
        # Mantém apenas as 10 leituras mais recentes
        if len(leituras) > 10:
            leituras.pop(0)
        
        # Calcula a média se já houver 10 leituras
        if len(leituras) == 10:
            media_c = sum(leituras) / len(leituras)              # Média das 10 leituras
            media_c = round(media_c, 1)                           # Arredonda para 1 casa decimal
            media_f = round((media_c * 9 / 5) + 32, 1)            # Converte para Fahrenheit (1 casa decimal)
            
            # Exibe no console
            print(f"Média das últimas 10 leituras: {media_c} °C | {media_f} °F")
        else:
            # Exibe leitura bruta (antes de atingir 10 amostras)
            print(f"Coletando leituras... ({len(leituras)}/10) Valor atual: {temp_c:.1f} °C")

    except Exception as e:
        # Caso ocorra erro de comunicação com o sensor
        print(f"Erro na leitura: {e}")

    # Aguarda 3 segundos antes da próxima leitura
    time.sleep(3)
