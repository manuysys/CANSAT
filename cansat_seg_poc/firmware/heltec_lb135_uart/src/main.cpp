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
 *  · GPS: este emisor NO tiene GPS; lat/lon son simulados y se declaran.
 *
 * Flasheo (PlatformIO):  pio run -e heltec_wifi_lora_32_V3 -t upload
 * Prueba en PC:          python uart_listener.py --port COMx --out outputs/uart_log.jsonl
 * Prueba en la Pi:       python uart_listener.py --port /dev/serial0   (GPIO15)
 */
#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_BME280.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

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
  Serial.println("# GPS: no hay -> lat/lon SIMULADOS (declarado)");
}

void loop() {
  float t_s = (float)pkt;  // 1 Hz
  float alt, p, temp, hum;

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

  float lat = LAT_BASE - (DERIVA_LAT_M_S * t_s) / 111320.0f;
  float lon = LON_BASE + (DERIVA_LON_M_S * t_s) / (111320.0f * 0.82f);

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

  // MPU6050: chequeo cada 5 paquetes, SOLO por serial (fuera del contrato).
  if (mpu_ok && (pkt % 5) == 0) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    Serial.print("# MPU6050 accel (m/s2): ");
    Serial.print(a.acceleration.x, 2);
    Serial.print(", ");
    Serial.print(a.acceleration.y, 2);
    Serial.print(", ");
    Serial.println(a.acceleration.z, 2);
  }

  pkt++;
  delay(1000);
}
