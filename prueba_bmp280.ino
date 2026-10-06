#include <Wire.h>
#include <Adafruit_BMP280.h>

#define I2C_SDA 41
#define I2C_SCL 42

Adafruit_BMP280 bmp;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Iniciando prueba del BMP280...");

  Wire.begin(I2C_SDA, I2C_SCL);

  bool encontrado = false;

  if (bmp.begin(0x76)) {
    Serial.println("BMP280 encontrado en 0x76");
    encontrado = true;
  }
  else if (bmp.begin(0x77)) {
    Serial.println("BMP280 encontrado en 0x77");
    encontrado = true;
  }

  if (!encontrado) {
    Serial.println("ERROR: No se encontro el BMP280");
    while (true);
  }

  Serial.println("BMP280 iniciado correctamente");
}

void loop() {
  float temperatura = bmp.readTemperature();
  float presion = bmp.readPressure() / 100.0F;
  float altura = bmp.readAltitude(1013.25);

  Serial.println("================================");

  Serial.print("Temperatura: ");
  Serial.print(temperatura);
  Serial.println(" °C");

  Serial.print("Presion: ");
  Serial.print(presion);
  Serial.println(" hPa");

  Serial.print("Altura estimada: ");
  Serial.print(altura);
  Serial.println(" m");

  delay(1000);
}