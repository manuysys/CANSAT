/*
 * Heltec WiFi LoRa 32 V3 / ESP8266 — emisor UART del contrato LB135 v2.
 *
 * Emite por USB-serie (115200 8N1) una línea por segundo con el mismo
 * formato que parsea uart_listener.py y emite mission_pipeline.py:
 *   $LB135,2,<pkt>,<t_s>,<alt_m>,<p_hPa>,<temp_C>,<veg>,<bui>,<wat>,<bare>,
 *   <oth>,<dom>,<usi>,<gvi>,<vcode>,<personas>,<vehiculos>,<alert>,
 *   <danado_pct>,<lat>,<lon>,<hum_pct>*HH
 * (checksum XOR estilo NMEA; ver cansat/protocol.py::format_packet).
 *
 * SENSORES (2026-09-26):
 *  · BME280 por I2C (Heltec V3: SDA=GPIO41, SCL=GPIO42; NodeMCU: D2/D1):
 *    si responde, p_hPa/temp_C/hum_pct son REALES y la altitud sale de la
 *    presión (atmósfera estándar). Si no, cae al perfil SIMULADO declarado.
 *  · MPU6050 en el mismo bus: se chequea (scan + acelerómetro) y se imprime
 *    por serial, pero NO viaja en el contrato v2 (el IMU es dominio del
 *    firmware de vuelo del equipo; ver decisiones `mpu-fuera-contrato`).
 *  · GPS (2026-10-05): soporta el ATGM336H real por UART2 (RX=GPIO6,
 *    TX=GPIO5 @9600, NMEA RMC). Con fix fresco (<5 s) el contrato lleva
 *    lat/lon REALES; sin fix cae al perfil simulado DECLARADO (como antes).
 *
 * ENLACE LoRa (2026-10-05): el radio del CanSat. `LORA_MODO`:
 *   0 = solo UART por cable (default; ESP8266 sin radio),
 *   1 = TX de VUELO: emite la misma línea LB135 por LoRa a 1 Hz (además del
 *       UART para la Pi),
 *   2 = RX de ESTACIÓN: recibe por LoRa y la re-emite por USB al PC (el
 *       listener y la estación leen el mismo protocolo, sin cambios).
 * Pines SX1262 de la Heltec V3: NSS=8, SCK=9, MOSI=10, MISO=11, RST=12,
 * BUSY=13, DIO1=14. Frecuencia 915 MHz (AU915), SF9, BW125, sync 0x12.
 *
 * Flasheo (PlatformIO):
 *   vuelo : pio run -e heltec_wifi_lora_32_V3 -t upload
 *   estación: pio run -e heltec_lora_rx -t upload
 * Prueba en PC:  python uart_listener.py --port COMx --out outputs/uart_log.jsonl
 * Prueba en la Pi: python uart_listener.py --port /dev/serial0   (GPIO15)
 */
#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_BME280.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ── Enlace LoRa: modo por env de PlatformIO (ver platformio.ini) ────────
#ifndef LORA_MODO
#define LORA_MODO 0
#endif
#ifndef LORA_FREC_MHZ
#define LORA_FREC_MHZ 915.0f
#endif
#ifndef LORA_POT_DBM
#define LORA_POT_DBM 14
#endif

#if LORA_MODO != 0 && defined(ARDUINO_ARCH_ESP32)
#include <RadioLib.h>
static const int LORA_NSS = 8, LORA_DIO1 = 14, LORA_RST = 12, LORA_BUSY = 13;
static SX1262 lora = new Module(LORA_NSS, LORA_DIO1, LORA_RST, LORA_BUSY);
static bool lora_ok = false;
static uint32_t lora_tx_n = 0, lora_rx_n = 0;

static void lora_iniciar() {
  Serial.print("# LoRa: ");
  Serial.print(LORA_MODO == 1 ? "TX" : "RX");
  Serial.print(" @ ");
  Serial.print(LORA_FREC_MHZ, 1);
  Serial.print(" MHz ... ");
  int st = lora.begin(LORA_FREC_MHZ, 125.0f, 9, 5, 0x12, LORA_POT_DBM, 8);
  if (st == RADIOLIB_ERR_NONE) {
    lora_ok = true;
    Serial.print("OK SF9 BW125 ");
    Serial.print(LORA_POT_DBM);
    Serial.println(" dBm");
  } else {
    Serial.print("init FALLO, codigo ");
    Serial.println(st);
  }
}
#endif

#if defined(ARDUINO_ARCH_ESP32)
// Heltec WiFi LoRa 32 V3: pines rotulados SDA/SCL del header (GPIO41/42).
// OJO: 17/18 son el I2C interno del OLED y NO están en el header.
static const int PIN_SDA = 41;
static const int PIN_SCL = 42;
#else
// NodeMCU/Wemos ESP8266: D2 = GPIO4 (SDA), D1 = GPIO5 (SCL).
static const int PIN_SDA = 4;
static const int PIN_SCL = 5;
#endif

// Perfil simulado (solo si no hay BME280): descenso 800 m a ~8 m/s.
static const float ALT_INICIAL_M = 800.0f;
static const float TASA_DESCENSO_M_S = 8.0f;
static const float LAT_BASE = -34.60372f;   // simulado (se declara)
static const float LON_BASE = -58.38159f;
static const float DERIVA_LAT_M_S = 0.6f;
static const float DERIVA_LON_M_S = 0.4f;

// ── GPS ATGM336H real (2026-10-05) ──────────────────────────────────────
// UART2 de la Heltec: RX=GPIO6 (desde el TX del GPS), TX=GPIO5, 9600 8N1.
// Si hay fix (RMC con status A y <5 s de antigüedad) el contrato lleva
// lat/lon REALES; si no, cae al perfil simulado DECLARADO (como antes).
#if defined(ARDUINO_ARCH_ESP32)
#define GPS_MODO 1
#else
#define GPS_MODO 0
#endif
#if GPS_MODO
static const int GPS_RX_PIN = 6, GPS_TX_PIN = 5;
static float gps_lat = 0.0f, gps_lon = 0.0f;
static bool gps_ok = false;
static uint32_t gps_ms = 0, gps_fix_n = 0;

static void gps_iniciar() {
  Serial2.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("# GPS: UART2 9600 listo (ATGM336H; fix real si llega)");
}

// ddmm.mmmm + hemisferio → grados decimales.
static float nmea_grados(const char *campo, char hemi) {
  float v = atof(campo);
  int grados = (int)(v / 100.0f);
  float minutos = v - (float)grados * 100.0f;
  float dec = (float)grados + minutos / 60.0f;
  if (hemi == 'S' || hemi == 'W') dec = -dec;
  return dec;
}

static void gps_leer() {
  static String buf;
  while (Serial2.available()) {
    char c = (char)Serial2.read();
    if (c == '\n') {
      buf.trim();
      if (buf.startsWith("$GPRMC") || buf.startsWith("$GNRMC")) {
        String tok[13];
        int idx = 0, start = 0;
        for (int i = 0; i <= (int)buf.length() && idx < 13; i++) {
          if (i == (int)buf.length() || buf[i] == ',') {
            tok[idx++] = buf.substring(start, i);
            start = i + 1;
          }
        }
        if (idx >= 7 && tok[2] == "A" && tok[3].length() && tok[5].length()) {
          gps_lat = nmea_grados(tok[3].c_str(), tok[4].length() ? tok[4][0] : 'N');
          gps_lon = nmea_grados(tok[5].c_str(), tok[6].length() ? tok[6][0] : 'E');
          gps_ok = true;
          gps_ms = millis();
          gps_fix_n++;
        }
      }
      buf = "";
    } else if (c != '\r') {
      buf += c;
      if (buf.length() > 120) buf = "";
    }
  }
}
#endif

static Adafruit_BME280 bme;
static Adafruit_MPU6050 mpu;
static bool bme_ok = false;
static bool mpu_ok = false;
static uint8_t bme_addr = 0;
static uint8_t mpu_addr = 0;
static uint32_t pkt = 0;

// XOR de los bytes del cuerpo (entre `$` y `*`), hex de 2 dígitos.
static void checksum_hex(const char *body, char *out) {
  uint8_t x = 0;
  for (const char *p = body; *p; p++) x ^= (uint8_t)(*p);
  const char *hex = "0123456789ABCDEF";
  out[0] = hex[(x >> 4) & 0xF];
  out[1] = hex[x & 0xF];
  out[2] = '\0';
}

static void escanear_i2c() {
  Serial.println("# scan I2C:");
  int n = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("#   dispositivo en 0x");
      Serial.println(addr, HEX);
      n++;
    }
  }
  if (n == 0) Serial.println("#   (ninguno: sensores no detectados)");
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Wire.begin(PIN_SDA, PIN_SCL);
  escanear_i2c();

  if (bme.begin(0x76, &Wire)) {
    bme_ok = true;
    bme_addr = 0x76;
  } else if (bme.begin(0x77, &Wire)) {
    bme_ok = true;
    bme_addr = 0x77;
  }
  if (mpu.begin(0x68, &Wire)) {
    mpu_ok = true;
    mpu_addr = 0x68;
  } else if (mpu.begin(0x69, &Wire)) {
    mpu_ok = true;
    mpu_addr = 0x69;
  }

#if LORA_MODO != 0 && defined(ARDUINO_ARCH_ESP32)
  lora_iniciar();
#endif

  Serial.println("# heltec_lb135_uart: emisor v2 listo");
  Serial.print("# BME280: ");
  if (bme_ok) {
    Serial.print("OK en 0x");
    Serial.print(bme_addr, HEX);
    Serial.println(" (p/t/hum REALES, alt desde presion)");
  } else {
    Serial.println("no detectado -> perfil SIMULADO");
  }
  Serial.print("# MPU6050: ");
  if (mpu_ok) {
    Serial.print("OK en 0x");
    Serial.print(mpu_addr, HEX);
    Serial.println(" (solo debug por serial, fuera del contrato)");
  } else {
    Serial.println("no detectado");
  }
#if GPS_MODO
  gps_iniciar();
#else
  Serial.println("# GPS: no hay -> lat/lon SIMULADOS (declarado)");
#endif
}

// ── U3: eventos de vuelo con el MPU6050 ──────────────────────────────────
// Máquina de estados con histéresis, evaluada 1×/s (el loop ya es 1 Hz):
//   EN_TIERRA → |a| se aparta de 1g N_MOV muestras seguidas → DESPEGUE
//   EN_VUELO  → |a| vuelve a ~1g N_QUIETO muestras seguidas → ATERRIZAJE
// Los eventos van por líneas debug "# EVENTO ..." (FUERA del contrato de
// radio: ver decisión `mpu-fuera-contrato`); el listener de la Pi las parsea
// y las propaga por uart_state.json → JSONL (`evento_vuelo`).
static bool en_vuelo = false;
static int n_mov = 0, n_quieto = 0;
static const float UMBRAL_MOV = 2.0f;    // m/s² de desvío de 1g
static const float UMBRAL_QUIETO = 0.6f;
static const int N_MOV = 3;              // 3 s de movimiento → despegue
static const int N_QUIETO = 15;          // 15 s quieto → aterrizaje

void chequear_evento_vuelo() {
  if (!mpu_ok) return;
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  float amag = sqrtf(a.acceleration.x * a.acceleration.x +
                     a.acceleration.y * a.acceleration.y +
                     a.acceleration.z * a.acceleration.z);
  float desvio = fabsf(amag - 9.81f);
  if (!en_vuelo) {
    n_mov = (desvio > UMBRAL_MOV) ? n_mov + 1 : 0;
    if (n_mov >= N_MOV) {
      en_vuelo = true;
      n_quieto = 0;
      Serial.print("# EVENTO DESPEGUE t_ms=");
      Serial.println(millis());
    }
  } else {
    n_quieto = (desvio < UMBRAL_QUIETO) ? n_quieto + 1 : 0;
    if (n_quieto >= N_QUIETO) {
      en_vuelo = false;
      n_mov = 0;
      Serial.print("# EVENTO ATERRIZAJE t_ms=");
      Serial.println(millis());
    }
  }
}

void loop() {
#if LORA_MODO == 2
  // RX de ESTACIÓN: recibe por LoRa y re-emite por USB el mismo protocolo.
  if (!lora_ok) {
    delay(1000);
    return;
  }
  String str;
  int st = lora.receive(str, 500);
  if (st == RADIOLIB_ERR_NONE) {
    lora_rx_n++;
    Serial.println(str);                 // el PC lo lee como al emisor cableado
    if (lora_rx_n % 10 == 1) {
      Serial.print("# LORA rx=");
      Serial.print(lora_rx_n);
      Serial.print(" rssi=");
      Serial.print(lora.getRSSI());
      Serial.print(" snr=");
      Serial.println(lora.getSNR());
    }
  } else if (st != RADIOLIB_ERR_RX_TIMEOUT) {
    Serial.print("# LORA rx error: ");
    Serial.println(st);
  }
#else
  float t_s = (float)pkt;  // 1 Hz
  float alt, p, temp, hum;

#if GPS_MODO
  gps_leer();
#endif

  if (bme_ok) {
    p = bme.readPressure() / 100.0f;          // Pa -> hPa
    temp = bme.readTemperature();
    hum = bme.readHumidity();
    // Atmósfera estándar: altitud a partir de la presión real.
    alt = 44330.0f * (1.0f - powf(p / 1013.25f, 0.1903f));
  } else {
    alt = ALT_INICIAL_M - TASA_DESCENSO_M_S * t_s;
    if (alt < 0) {  // fin del descenso simulado: reinicia el perfil
      pkt = 0;
      Serial.println("# perfil reiniciado");
      delay(1000);
      return;
    }
    p = 1013.25f * powf(1.0f - 2.25577e-5f * alt, 5.25588f);
    temp = 20.0f - 0.0065f * alt;
    hum = 55.0f;
  }

  // Mezcla de terreno: con sensor real queda estática (altitud de escritorio);
  // sin sensor sigue el descenso simulado.
  float f = 1.0f - alt / ALT_INICIAL_M;
  if (f < 0.0f) f = 0.0f;
  if (f > 1.0f) f = 1.0f;
  float veg = 78.0f - 66.0f * f;
  float bui = 1.0f + 54.0f * f;
  float wat = 6.0f - 4.0f * f;
  float bare = 12.0f + 8.0f * f;
  float oth = 3.0f + 8.0f * f;
  int dom = 0;
  float mx = veg;
  const float vals[5] = {veg, bui, wat, bare, oth};
  for (int i = 1; i < 5; i++) {
    if (vals[i] > mx) {
      mx = vals[i];
      dom = i;
    }
  }
  float usi = bui / (veg > 0.1f ? veg : 0.1f);
  float gvi = (veg - bare) / (veg + bare > 1.0f ? veg + bare : 1.0f);

  float lat, lon;
#if GPS_MODO
  // Fix real si es fresco (<5 s); si no, perfil simulado declarado.
  bool gps_fresco = gps_ok && (millis() - gps_ms) < 5000;
  if (gps_fresco) {
    lat = gps_lat;
    lon = gps_lon;
  } else {
    lat = LAT_BASE - (DERIVA_LAT_M_S * t_s) / 111320.0f;
    lon = LON_BASE + (DERIVA_LON_M_S * t_s) / (111320.0f * 0.82f);
  }
  if (pkt % 30 == 0) {
    Serial.print("# GPS fix=");
    Serial.print(gps_fresco ? 1 : 0);
    Serial.print(" n=");
    Serial.print(gps_fix_n);
    if (gps_fresco) {
      Serial.print(" lat=");
      Serial.print(lat, 5);
      Serial.print(" lon=");
      Serial.print(lon, 5);
    }
    Serial.println();
  }
#else
  lat = LAT_BASE - (DERIVA_LAT_M_S * t_s) / 111320.0f;
  lon = LON_BASE + (DERIVA_LON_M_S * t_s) / (111320.0f * 0.82f);
#endif

  char body[192];
  snprintf(body, sizeof(body),
           "LB135,2,%lu,%07.1f,%06.1f,%06.1f,%05.1f,"
           "%5.1f,%5.1f,%5.1f,%5.1f,%5.1f,"
           "%d,%6.2f,%+6.3f,%d,%d,%d,%d,"
           "%5.1f,%.5f,%.5f,%4.1f",
           (unsigned long)pkt, t_s, alt, p, temp, veg, bui, wat, bare, oth,
           dom, usi, gvi, 0, 0, 0, 0, 0.0f, lat, lon, hum);
  char cs[4];
  checksum_hex(body, cs);
  Serial.print('$');
  Serial.print(body);
  Serial.print('*');
  Serial.println(cs);

#if LORA_MODO == 1
  // TX de vuelo: la MISMA línea LB135 viaja por LoRa (el contrato no cambia).
  if (lora_ok) {
    char linea[200];
    snprintf(linea, sizeof(linea), "$%s*%s", body, cs);
    int st = lora.transmit(linea);
    if (st == RADIOLIB_ERR_NONE) {
      lora_tx_n++;
      if (pkt % 10 == 0) {
        Serial.print("# LORA tx=");
        Serial.print(lora_tx_n);
        Serial.print(" rssi=");
        Serial.println(lora.getRSSI());
      }
    } else {
      Serial.print("# LORA tx error: ");
      Serial.println(st);
    }
  }
#endif

  // MPU6050: eventos de vuelo (U3) cada segundo + debug cada 5 paquetes.
  // Todo por serial, fuera del contrato de radio.
  chequear_evento_vuelo();
  if (mpu_ok && (pkt % 5) == 0) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    Serial.print("# MPU6050 accel (m/s2): ");
    Serial.print(a.acceleration.x, 2);
    Serial.print(", ");
    Serial.print(a.acceleration.y, 2);
    Serial.print(", ");
    Serial.print(a.acceleration.z, 2);
    Serial.print(" | en_vuelo=");
    Serial.println(en_vuelo ? 1 : 0);
  }

  pkt++;
  delay(1000);
#endif  // LORA_MODO != 2
}
