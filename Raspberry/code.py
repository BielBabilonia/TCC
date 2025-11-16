# ============================================================
# DS18B20 -> CSV via Serial (CircuitPython / Thonny)
# MODO "cada": imprime 1 leitura por segundo
# MODO "janela": média das leituras a cada JANELA_S segundos
# ============================================================

import time
import board
import wifi
import socketpool
import ssl
import adafruit_requests
import adafruit_onewire.bus
import adafruit_ds18x20
import rtc
import adafruit_ntp

# ---------- MODO ----------
MODO = "cada"   # "cada" ou "janela"
JANELA_S = 5    # usado só no modo "janela"
INTERVALO_S = 1 # intervalo entre leituras (modo "cada")

# ---------- CONFIG Wi-Fi ----------
WIFI_SSID = "eegabriell"
WIFI_PASS = "gaab1234"

# ---------- SENSOR ----------
ow_pin = board.GP5
ow_bus = adafruit_onewire.bus.OneWireBus(ow_pin)
devices = ow_bus.scan()
if not devices:
    raise RuntimeError("Nenhum sensor DS18B20 encontrado!")
ds18 = adafruit_ds18x20.DS18X20(ow_bus, devices[0])

# ---------- (OPCIONAL) ENVIO PARA GOOGLE ----------
ENVIAR_GOOGLE = False
GOOGLE_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbwzK09qU4kHyC8-iqgozHlIKtUBvSUEJjouNn5DrDceG19JWbeE_wmtLX8QNonDeKR4/exec"

# ---------- FUSO ----------
FUSO_OFFSET_HORAS = -3  # UTC-3

# ---------- CONECTA Wi-Fi + NTP ----------
print("Conectando ao Wi-Fi...")
wifi.radio.connect(WIFI_SSID, WIFI_PASS)
print("Wi-Fi conectado! IP:", wifi.radio.ipv4_address)

pool = socketpool.SocketPool(wifi.radio)
ssl_context = ssl.create_default_context()
requests = adafruit_requests.Session(pool, ssl_context)

print("Sincronizando NTP...")
try:
    ntp = adafruit_ntp.NTP(pool, server="pool.ntp.org")
    rtc.RTC().datetime = ntp.datetime
    print("Hora sincronizada via NTP.")
except Exception as e:
    print("Falha no NTP:", e)

def timestamp_iso():
    agora = time.localtime(time.time() + FUSO_OFFSET_HORAS * 3600)
    return "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}".format(*agora[:6])

print("timestamp_iso,temperatura_c")  # cabeçalho CSV

# ---------- LOOP ----------
if MODO == "cada":
    while True:
        try:
            temp_c = round(ds18.temperature, 2)  # ~750 ms de conversão
            print(f"{timestamp_iso()},{temp_c}")

            if ENVIAR_GOOGLE:
                try:
                    data = f"temperatura={temp_c}&timestamp={timestamp_iso()}"
                    r = requests.post(GOOGLE_WEBAPP_URL, data=data,
                                      headers={"Content-Type": "application/x-www-form-urlencoded"})
                    # print("Google:", r.text)
                except Exception as e:
                    print("Erro ao enviar:", e)

            time.sleep(INTERVALO_S)  # ajuste se quiser mais/menos rápido
        except Exception as e:
            print("Erro no loop:", e)
            time.sleep(1)

elif MODO == "janela":
    buffer = []
    t0 = time.monotonic()
    while True:
        try:
            temp_c = round(ds18.temperature, 2)
            buffer.append(temp_c)

            if time.monotonic() - t0 >= JANELA_S:
                media_c = round(sum(buffer) / len(buffer), 2)
                ts = timestamp_iso()
                print(f"{ts},{media_c}")

                if ENVIAR_GOOGLE:
                    try:
                        data = f"temperatura={media_c}&timestamp={ts}"
                        r = requests.post(GOOGLE_WEBAPP_URL, data=data,
                                          headers={"Content-Type": "application/x-www-form-urlencoded"})
                        # print("Google:", r.text)
                    except Exception as e:
                        print("Erro ao enviar:", e)

                buffer.clear()
                t0 = time.monotonic()

            time.sleep(0.5)  # lê mais vezes dentro da janela
        except Exception as e:
            print("Erro no loop:", e)
            time.sleep(1)
else:
    raise ValueError("MODO inválido: use 'cada' ou 'janela'")