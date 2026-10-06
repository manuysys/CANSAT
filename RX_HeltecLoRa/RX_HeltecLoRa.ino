/*******************************************************************
 * Receptor RX corregido para Heltec WiFi LoRa 32 V3
 *
 * IMPORTANTE:
 * - No se usa VEXT.
 * - Parseo robusto con sscanf.
 * - Agrega estadísticas de paquetes recibidos, perdidos y resets.
 * - Mantiene compatibilidad con formato:
 *
 * pkt, presionBase_hPa, presionActual_hPa,
 * altura_m, temperatura_C, humedad_%
 *******************************************************************/

#include <RadioLib.h>
#include <SPI.h>

char Version[] = "v4.0 RX-HeltecV3-BME280";

// ------------------------------
// Pines Heltec V3
// ------------------------------

#define HELTEC_NSS 8
#define HELTEC_DIO1 14
#define HELTEC_RST 12
#define HELTEC_BUSY 13
#define LED 35

// Si en algún momento usan OLED u otro periférico VEXT,
// pueden agregar y habilitar VEXT aparte.
// #define VEXT_PIN 36

#define SERIAL_BAUDRATE 115200

// ------------------------------
// Configuración LoRa
// ------------------------------

#define LORA_FREQUENCY 915.0
#define LORA_BANDWIDTH 125.0
#define LORA_SPREAD_FACTOR 7
#define LORA_CODING_RATE 5
#define LORA_SYNC_WORD 0xF3

// ------------------------------
// Objeto LoRa
// ------------------------------

SX1262 radio = new Module(
  HELTEC_NSS,
  HELTEC_DIO1,
  HELTEC_RST,
  HELTEC_BUSY
);

// ------------------------------
// Estadísticas
// ------------------------------

unsigned long lastPacketNumber = 0;

unsigned long receivedCount = 0;
unsigned long lostCount = 0;
unsigned long resetCount = 0;
unsigned long parseErrorCount = 0;

// ------------------------------
// SETUP
// ------------------------------

void setup()
{
  pinMode(LED, OUTPUT);

  digitalWrite(LED, HIGH);

  Serial.begin(SERIAL_BAUDRATE);

  delay(1000);

  Serial.println();

  Serial.println(
    "--- Inicializando Receptor LoRa SX1262 ---"
  );

  Serial.print("Version: ");
  Serial.println(Version);

  // ------------------------------
  // Inicializar LoRa
  // ------------------------------

  int state = radio.begin(
    LORA_FREQUENCY,
    LORA_BANDWIDTH,
    LORA_SPREAD_FACTOR,
    LORA_CODING_RATE,
    LORA_SYNC_WORD
  );

  if (state == RADIOLIB_ERR_NONE)
  {
    Serial.println(
      "LoRa RX Inicializado Correctamente!"
    );

    digitalWrite(LED, LOW);
  }
  else
  {
    Serial.print(
      "Error al iniciar LoRa, codigo: "
    );

    Serial.println(state);

    while (true)
    {
      digitalWrite(
        LED,
        !digitalRead(LED)
      );

      delay(200);
    }
  }
}

// ------------------------------
// LOOP
// ------------------------------

void loop()
{
  String strData;

  int state = radio.receive(strData);

  if (state != RADIOLIB_ERR_NONE)
  {
    // Sin paquete recibido.
    // No imprimimos error para no llenar
    // el puerto serie con timeouts.

    return;
  }

  digitalWrite(LED, HIGH);

  float rssi = radio.getRSSI();

  float snr = radio.getSNR();

  char buffer[128];

  strData.toCharArray(
    buffer,
    sizeof(buffer)
  );

  // ------------------------------
  // Variables recibidas
  // ------------------------------

  unsigned long pktNumber = 0;

  float presionBase = 0.0f;

  float presionAbsoluta = 0.0f;

  float altura = 0.0f;

  float temperatura = 0.0f;

  float humedad = 0.0f;

  // ------------------------------
  // Parsear paquete
  // ------------------------------

  int parsedFields = sscanf(
    buffer,
    "%lu,%f,%f,%f,%f,%f",
    &pktNumber,
    &presionBase,
    &presionAbsoluta,
    &altura,
    &temperatura,
    &humedad
  );

  Serial.println(
    "====================================================================="
  );

  Serial.print("RX - Version ");

  Serial.println(Version);

  Serial.print("Telemetria RAW: ");

  Serial.println(strData);

  // ------------------------------
  // Comprobar formato
  // ------------------------------

  if (parsedFields != 6)
  {
    parseErrorCount++;

    Serial.println(
      "ERROR: paquete recibido pero formato incorrecto."
    );

    Serial.printf(
      "Campos parseados: %d\n",
      parsedFields
    );
  }
  else
  {
    receivedCount++;

    // ------------------------------
    // Detección de pérdida de paquetes
    // ------------------------------

    if (lastPacketNumber != 0)
    {
      if (pktNumber > lastPacketNumber + 1)
      {
        lostCount +=
          (pktNumber - lastPacketNumber - 1);
      }
      else if (pktNumber < lastPacketNumber)
      {
        // El transmisor probablemente se reinició.

        resetCount++;
      }
    }

    lastPacketNumber = pktNumber;

    // ------------------------------
    // Mostrar telemetría
    // ------------------------------

    Serial.printf(
      "Packet Number: %lu\n",
      pktNumber
    );

    Serial.printf(
      "Presion Base: %.2f hPa\n",
      presionBase
    );

    Serial.printf(
      "Presion Absoluta: %.2f hPa\n",
      presionAbsoluta
    );

    Serial.printf(
      "Altura: %.2f m\n",
      altura
    );

    Serial.printf(
      "Temperatura: %.1f C\n",
      temperatura
    );

    Serial.printf(
      "Humedad: %.1f %%\n",
      humedad
    );

    Serial.printf(
      "RSSI: %.1f dBm\n",
      rssi
    );

    Serial.printf(
      "SNR: %.2f dB\n",
      snr
    );

    Serial.println(
      "------------------------------------"
    );

    Serial.printf(
      "Stats RX -> Recibidos: %lu | Perdidos: %lu | Resets TX: %lu | Errores parseo: %lu\n",
      receivedCount,
      lostCount,
      resetCount,
      parseErrorCount
    );
  }

  digitalWrite(LED, LOW);
}