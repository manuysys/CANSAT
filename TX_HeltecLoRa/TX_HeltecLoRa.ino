/*******************************************************************
 * Transmisor TX corregido para Heltec WiFi LoRa 32 V3 + BME280
 *
 * IMPORTANTE:
 * - BME280 alimentado desde VCC / 3V3.
 * - No se usa VEXT para alimentar el BME280.
 * - Usa presión de lanzamiento como referencia.
 * - Calcula altura relativa al punto de lanzamiento.
 * - Mantiene formato oficial:
 * pkt, presionBase_hPa, presionActual_hPa, altura_m,
 * temperatura_C, humedad_%
 *******************************************************************/

#include <RadioLib.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_BME280.h>

char Version[] = "v4.0 TX-BME280-HeltecV3-LaunchAlt-VCC";

// ------------------------------
// Pines Heltec V3
// ------------------------------

#define HELTEC_NSS 8
#define HELTEC_DIO1 14
#define HELTEC_RST 12
#define HELTEC_BUSY 13
#define LED 35

// Si en algún momento vuelven a usar la OLED u otro periférico
// en VEXT, pueden volver a habilitar VEXT manualmente.
// #define VEXT_PIN 36

// ------------------------------
// Pines I2C BME280
// ------------------------------

#define I2C_SDA 41
#define I2C_SCL 42

// ------------------------------
// Puerto serie
// ------------------------------

#define SERIAL_BAUDRATE 115200

// ------------------------------
// Intervalo de transmisión
// ------------------------------

#define INTERVAL_TIME_TX 1000

// ------------------------------
// Configuración LoRa
// ------------------------------

#define LORA_FREQUENCY 915.0
#define LORA_BANDWIDTH 125.0
#define LORA_SPREAD_FACTOR 7
#define LORA_CODING_RATE 5
#define LORA_SYNC_WORD 0xF3
#define LORA_POWER 17

// ------------------------------
// Configuración de altura
// ------------------------------

// 1 = altura relativa respecto al punto de lanzamiento.
// 0 = altura absoluta usando SEA_LEVEL_HPA.

#define USE_LAUNCH_PRESSURE 1

// Si USE_LAUNCH_PRESSURE = 0, se usa esta presión como nivel del mar.
#define SEA_LEVEL_HPA 1013.25

// ------------------------------
// Calibración de presión base
// ------------------------------

#define WARMUP_READINGS 15
#define BASE_SAMPLES 30
#define BASE_DELAY_MS 10

// ------------------------------
// Filtro de presión
// ------------------------------

// 1 = activa filtro suave de presión.
// 0 = usa presión directa del sensor.

#define ENABLE_PRESSURE_FILTER 1
#define PRESSURE_FILTER_ALPHA 0.35f

// ------------------------------
// Offset opcional de presión
// ------------------------------

#define PRESSURE_OFFSET_HPA 0.0f

// ------------------------------
// Objetos
// ------------------------------

SX1262 radio = new Module(
  HELTEC_NSS,
  HELTEC_DIO1,
  HELTEC_RST,
  HELTEC_BUSY
);

Adafruit_BME280 bme;

// ------------------------------
// Variables globales
// ------------------------------

unsigned long pktNumber = 0;

float baseline_hPa = 0.0f;
float pressureFiltered_hPa = 0.0f;
bool pressureFilterInitialized = false;

uint32_t lastTxMillis = 0;

// ------------------------------
// Funciones auxiliares
// ------------------------------

float pressurePaToHpa(float pressurePa)
{
  return pressurePa * 0.01f;
}

bool validPressureHpa(float hPa)
{
  if (isnan(hPa))
    return false;

  // Rango razonable para pruebas terrestres / CanSat.
  if (hPa < 200.0f || hPa > 1200.0f)
    return false;

  return true;
}

float readPressureHpa()
{
  float pressurePa = bme.readPressure();

  float pressureHpa = pressurePaToHpa(pressurePa);

  pressureHpa += PRESSURE_OFFSET_HPA;

  return pressureHpa;
}

float readPressureFilteredHpa()
{
  float p = readPressureHpa();

  if (!validPressureHpa(p))
  {
    // Si la lectura es inválida, devolvemos
    // el último valor filtrado válido.
    return pressureFilterInitialized
             ? pressureFiltered_hPa
             : 0.0f;
  }

#if ENABLE_PRESSURE_FILTER

  if (!pressureFilterInitialized)
  {
    pressureFiltered_hPa = p;
    pressureFilterInitialized = true;
  }
  else
  {
    pressureFiltered_hPa =
      (PRESSURE_FILTER_ALPHA * p) +
      ((1.0f - PRESSURE_FILTER_ALPHA) * pressureFiltered_hPa);
  }

  return pressureFiltered_hPa;

#else

  return p;

#endif
}

float calcAltitude(float pressure_hPa, float reference_hPa)
{
  if (pressure_hPa <= 0.0f || reference_hPa <= 0.0f)
  {
    return 0.0f;
  }

  // Fórmula barométrica estándar.
  //
  // Para altura relativa:
  // reference_hPa = presión de lanzamiento.
  //
  // Para altura absoluta:
  // reference_hPa = presión a nivel del mar local / QNH.

  return 44330.0f *
         (1.0f -
          pow(
            pressure_hPa / reference_hPa,
            0.190294f
          ));
}

void warmupBME280()
{
  for (int i = 0; i < WARMUP_READINGS; i++)
  {
    bme.readTemperature();
    bme.readPressure();
    bme.readHumidity();

    delay(30);
  }
}

float calibrateLaunchPressure()
{
  double sum = 0.0;
  int validCount = 0;

  for (int i = 0; i < BASE_SAMPLES; i++)
  {
    float p = readPressureHpa();

    if (validPressureHpa(p))
    {
      sum += p;
      validCount++;
    }

    delay(BASE_DELAY_MS);
  }

  if (validCount == 0)
  {
    return 0.0f;
  }

  return (float)(sum / validCount);
}

void readTelemetry(
  float &temperatureC,
  float &pressureHpa,
  float &altitudeM,
  float &humidity
)
{
  temperatureC = bme.readTemperature();

  pressureHpa = readPressureFilteredHpa();

  humidity = bme.readHumidity();

#if USE_LAUNCH_PRESSURE

  float reference_hPa = baseline_hPa;

#else

  float reference_hPa = SEA_LEVEL_HPA;

#endif

  altitudeM = calcAltitude(
    pressureHpa,
    reference_hPa
  );
}

void buildPacket(
  char *buffer,
  size_t bufferSize,
  float temperatureC,
  float pressureHpa,
  float altitudeM,
  float humidity
)
{
  snprintf(
    buffer,
    bufferSize,
    "%lu,%.2f,%.2f,%.2f,%.1f,%.1f",
    (unsigned long)pktNumber,
    baseline_hPa,
    pressureHpa,
    altitudeM,
    temperatureC,
    humidity
  );
}

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
  Serial.println("--- Inicializando Transmisor CanSat ---");

  Serial.print("Version: ");
  Serial.println(Version);

  Serial.println("BME280 alimentado desde VCC / 3V3.");

  // ------------------------------
  // Inicializar LoRa
  // ------------------------------

  int state = radio.begin(
    LORA_FREQUENCY,
    LORA_BANDWIDTH,
    LORA_SPREAD_FACTOR,
    LORA_CODING_RATE,
    LORA_SYNC_WORD,
    LORA_POWER
  );

  if (state == RADIOLIB_ERR_NONE)
  {
    Serial.println("LoRa inicializado correctamente.");
  }
  else
  {
    Serial.print("Error al iniciar LoRa, codigo: ");
    Serial.println(state);

    while (true)
    {
      digitalWrite(LED, !digitalRead(LED));
      delay(200);
    }
  }

  // ------------------------------
  // Inicializar I2C
  // ------------------------------

  Wire.begin(I2C_SDA, I2C_SCL);

  // ------------------------------
  // Inicializar BME280
  // ------------------------------

  bool bmeInit = false;

  while (!bmeInit)
  {
    if (bme.begin(0x76) || bme.begin(0x77))
    {
      bmeInit = true;
    }
    else
    {
      Serial.println(
        "BME280 no encontrado en GPIO41/GPIO42. Reintentando..."
      );

      delay(1000);
    }
  }

  // Configuración equivalente a la utilizada
  // anteriormente en el BMP280, agregando humedad.

  bme.setSampling(
    Adafruit_BME280::MODE_NORMAL,
    Adafruit_BME280::SAMPLING_X16,
    Adafruit_BME280::SAMPLING_X16,
    Adafruit_BME280::SAMPLING_X16,
    Adafruit_BME280::FILTER_X16,
    Adafruit_BME280::STANDBY_MS_500
  );

  Serial.println("BME280 encontrado e iniciado OK.");

  // ------------------------------
  // Calentamiento del sensor
  // ------------------------------

  Serial.println("Calentando BME280...");

  warmupBME280();

  // ------------------------------
  // Calibración de presión de lanzamiento
  // ------------------------------

  Serial.println("Calibrando presion de lanzamiento...");
  Serial.println(
    "IMPORTANTE: la placa debe estar quieta en el punto de lanzamiento."
  );

  baseline_hPa = calibrateLaunchPressure();

  if (!validPressureHpa(baseline_hPa))
  {
    Serial.println("ERROR: presion de lanzamiento invalida.");

    Serial.println(
      "Revisar BME280, cableado, alimentación y soldaduras."
    );

    while (true)
    {
      digitalWrite(LED, !digitalRead(LED));
      delay(200);
    }
  }

  // Inicializamos el filtro con la presión base.

  pressureFiltered_hPa = baseline_hPa;
  pressureFilterInitialized = true;

  Serial.printf(
    "Presion de lanzamiento: %.2f hPa\n",
    baseline_hPa
  );

#if USE_LAUNCH_PRESSURE

  Serial.println(
    "Modo de altura: RELATIVO al punto de lanzamiento."
  );

#else

  Serial.printf(
    "Modo de altura: ABSOLUTO usando nivel del mar: %.2f hPa\n",
    SEA_LEVEL_HPA
  );

#endif

  digitalWrite(LED, LOW);

  lastTxMillis = millis();
}

// ------------------------------
// LOOP
// ------------------------------

void loop()
{
  uint32_t now = millis();

  if (now - lastTxMillis < INTERVAL_TIME_TX)
  {
    return;
  }

  lastTxMillis = now;

  pktNumber++;

  float temperatureC = 0.0f;
  float pressureHpa = 0.0f;
  float altitudeM = 0.0f;
  float humidity = 0.0f;

  readTelemetry(
    temperatureC,
    pressureHpa,
    altitudeM,
    humidity
  );

  char packetData[96];

  buildPacket(
    packetData,
    sizeof(packetData),
    temperatureC,
    pressureHpa,
    altitudeM,
    humidity
  );

  Serial.println(
    "====================================================================="
  );

  Serial.printf(
    "TX - Version %s\n",
    Version
  );

  Serial.printf(
    "Packet Number: %lu\n",
    (unsigned long)pktNumber
  );

  Serial.printf(
    "Presion Base: %.2f hPa\n",
    baseline_hPa
  );

  Serial.printf(
    "Presion Actual: %.2f hPa\n",
    pressureHpa
  );

  Serial.printf(
    "Altura: %.2f m\n",
    altitudeM
  );

  Serial.printf(
    "Temperatura: %.1f C\n",
    temperatureC
  );

  Serial.printf(
    "Humedad: %.1f %%\n",
    humidity
  );

  Serial.printf(
    "Enviando: %s\n",
    packetData
  );

  digitalWrite(LED, HIGH);

  int state = radio.transmit(packetData);

  digitalWrite(LED, LOW);

  if (state == RADIOLIB_ERR_NONE)
  {
    Serial.println("Paquete enviado con exito.");
  }
  else
  {
    Serial.print("Error al transmitir: ");
    Serial.println(state);
  }
}