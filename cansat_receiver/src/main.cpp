#include <Arduino.h>

// RX-only firmware for Heltec WiFi LoRa 32 V3 (ESP32-S3)
// Config: SX1262, 915 MHz AU915, SF9, BW125, sync 0xF3
// NO transmits — only receives and prints LB135 telemetry

#include <RadioLib.h>

#define HELTEC_NSS 8
#define HELTEC_DIO1 14
#define HELTEC_RST 12
#define HELTEC_BUSY 13
#define LED_PIN 35

SX1262 radio = new Module(HELTEC_NSS, HELTEC_DIO1, HELTEC_RST, HELTEC_BUSY);

unsigned long pkt_count = 0;
unsigned long rx_count = 0;
unsigned long fail_count = 0;
unsigned long prev_pkt = 0;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("=== RX HeltecLoRa — SXR-only v4.0 ===");
  
  // begin(freq, BW, SF, CR, syncWord, power, gain)
  int st = radio.begin(915.0, 125.0, 9, 5, 0xF3, 17, 8);
  if (st == 0) {
    Serial.println("LoRa RX OK: 915 MHz SF9 BW125 sync=0xF3");
    digitalWrite(LED_PIN, LOW);
  } else {
    Serial.printf("LoRa init FAILED: %d\n", st);
  }
}

void loop() {
  String strData;
  int state = radio.receive(strData);
  if (state != 0) {
    fail_count++;
    return;
  }
  
  rx_count++;
  dtostrf(radio.getRSSI(), 1, 2, nullptr);
  dtostrf(radio.getSNR(), 1, 2, nullptr);
  
  char rssi_s[8], snr_s[8];
  dtostrf(radio.getRSSI(), 1, 2, rssi_s);
  dtostrf(radio.getSNR(), 1, 2, snr_s);
  
  Serial.printf("### RX #%lu rssi=%s snr=%s\n", rx_count, rssi_s, snr_s);
  Serial.printf("RAW: %s\n", strData.c_str());
  
  // Try to parse LB135 fragment: extract packet number (first number)
  char* pEnd;
  unsigned long pn = strtoul(strData.c_str(), &pEnd, 10);
  if (pn > prev_pkt + 1 && prev_pkt > 0) {
    Serial.printf("LOST: %lu packets\n", pn - prev_pkt - 1);
  }
  prev_pkt = pn;
}
