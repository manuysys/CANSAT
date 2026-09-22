/*
 * Heltec WiFi LoRa 32 V3 — emisor UART del contrato LB135 v2 (PERFIL SIMULADO).
 *
 * Emite por USB-serie (Serial, 115200 8N1) una línea por segundo con el mismo
 * formato que parsea uart_listener.py y emite mission_pipeline.py:
 *   $LB135,2,<pkt>,<t_s>,<alt_m>,<p_hPa>,<temp_C>,<veg>,<bui>,<wat>,<bare>,
 *   <oth>,<dom>,<usi>,<gvi>,<vcode>,<personas>,<vehiculos>,<alert>,
 *   <danado_pct>,<lat>,<lon>,<hum_pct>*HH
 * (checksum XOR estilo NMEA; ver cansat/protocol.py::format_packet).
 *
 * Qué valida y qué NO:
 *  · SÍ valida el contrato de punta a punta con hardware real (Heltec →
 *    uart_listener.py → uart_state.json → mission_pipeline --uart-state).
 *  · NO valida el enlace GPIO (TX→RX directo Pi↔ESP32): sale por USB-CDC, así
 *    que en la Pi se lee como /dev/ttyACM0. El cableado GPIO15 queda pendiente
 *    y declarado (ver pi/guia_pi.md §8 y decisiones.yaml `uart-heltec`).
 *  · Los SENSORES son simulados (descenso coherente, no BME280/GPS reales):
 *    sirve para probar el protocolo, no la instrumentación.
 *
 * Flasheo (PlatformIO):  pio run -t upload
 * Prueba en PC:          python uart_listener.py --port COMx --out outputs/uart_log.jsonl
 * Prueba en la Pi:       python uart_listener.py --port /dev/ttyACM0
 */
#include <Arduino.h>

// Perfil simulado: descenso de 800 m a ~8 m/s, 1 paquete/s, luego se repite.
static const float ALT_INICIAL_M = 800.0f;
static const float TASA_DESCENSO_M_S = 8.0f;
static const float LAT_BASE = -34.60372f;   // fijado simulado (se declara)
static const float LON_BASE = -58.38159f;
static const float DERIVA_LAT_M_S = 0.6f;   // ~m/s hacia el sur
static const float DERIVA_LON_M_S = 0.4f;   // ~m/s hacia el este

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

void setup() {
  Serial.begin(115200);
  // En S3 el USB-CDC tarda en enumerar; sin esto se pierden los primeros.
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);
  Serial.println("# heltec_lb135_uart: emisor v2 (SIMULADO) listo");
}

void loop() {
  float t_s = (float)pkt;  // 1 Hz
  float alt = ALT_INICIAL_M - TASA_DESCENSO_M_S * t_s;
  if (alt < 0) {  // fin del descenso: reinicia el perfil (se declara en el log)
    pkt = 0;
    Serial.println("# perfil reiniciado");
    delay(1000);
    return;
  }

  // Atmósfera estándar (misma que sim_uart.py).
  float p = 1013.25f * powf(1.0f - 2.25577e-5f * alt, 5.25588f);
  float temp = 20.0f - 0.0065f * alt;

  // Mezcla de terreno rural→urbana a medida que baja (perfil coherente).
  float f = 1.0f - alt / ALT_INICIAL_M;  // 0 arriba, 1 abajo
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
           dom, usi, gvi, 0, 0, 0, 0, 0.0f, lat, lon, 55.0f);
  char cs[4];
  checksum_hex(body, cs);
  Serial.print('$');
  Serial.print(body);
  Serial.print('*');
  Serial.println(cs);

  pkt++;
  delay(1000);
}
