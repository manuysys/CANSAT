# Electrónica y PCB — conexionado Pi ↔ Heltec ↔ sensores

Guía de diseño para la placa del CanSat "La Base" (CONAE 135). Responde a las
preguntas de integración entre la **Raspberry Pi Zero W** (misión secundaria) y
la **Heltec WiFi LoRa 32 V3** (computadora de vuelo + radio), según el DPD.

---

## 1. Qué necesita la Pi (resumen)

| Conexión | ¿Obligatoria? | Por qué |
|---|---|---|
| **5V + GND** desde la placa | Sí | Alimentación (el step-up de 5V) |
| **UART con la Heltec** | **Sí** | Es el único camino por el que la Pi recibe hora, GPS, presión y temperatura para etiquetar cada imagen (DPD). Lo implementa `uart_listener.py` + `mission_pipeline.py --uart-state` |
| **Cámara CSI** | Sí | Va al conector CSI de la Pi (cable **mini 22 pines**), **nunca** a la Heltec |
| LoRa (antena) | No es de la Pi | El radio lo maneja la Heltec; la Pi solo habla UART |

> La Pi **no** se conecta directo al módulo LoRa ni a los sensores: todo pasa
> por la Heltec. La Pi es un consumidor de telemetría + cámara + cómputo.

## 1.1 Inventario de componentes (DPD + realidad)

| Componente | Dónde va | Conexión |
|---|---|---|
| Raspberry Pi Zero W v1 (DPD: Zero 2 W) | PCB de vuelo | Header 40 pines: 5V (2/4), GND (6), TX (8), RX (10) |
| AI Camera IMX500 (DPD: Camera Module 3) | Pi, conector CSI | Cable mini 22 pines ↔ cámara 15 pines; **no** toca la PCB |
| microSD | Pi | Ranura de la Pi; asegurar contra vibración |
| Heltec WiFi LoRa 32 V3 **#1** (vuelo) | PCB de vuelo | J2/J3: 5V, GND, UART a la Pi, I2C/UART a sensores, antena IPEX |
| Heltec WiFi LoRa 32 V3 **#2** (terrena) | Estación terrena | USB-C a la notebook; **no necesita PCB** |
| GPS ATGM336H (Neo-M8N) | PCB de vuelo | UART2 a la Heltec #1 (3.3 V) |
| BME280 | PCB de vuelo | I2C a la Heltec #1 (3.3 V) |
| MPU6050 | PCB de vuelo | I2C a la Heltec #1 (3.3 V) |
| Batería 18650 (3.7 V) | PCB de vuelo | Al módulo XY-J02 |
| XY-J02 (carga de 18650) | PCB de vuelo | Entrada USB para cargar; salida al step-up |
| Step-Up 5V (DCDC-0.9-5-STU) | PCB de vuelo | Riel de 5V → Pi + Heltec #1 |
| LM317 (DPD) | — | **Redundante** para carga (el XY-J02 ya carga). Se puede eliminar con justificación (ahorra 4 g) |

## 2. Alimentación (5V)

```
BATERÍA Li-ion 18650 (3.7V)
      │
  [XY-J02] ──► carga/protección (USB para cargar)
      │
  [Step-Up 5V] ──► riel 5V de la PCB
      │
      ├──► Pi Zero W:  5V (pin 2 o 4 del GPIO) + GND
      └──► Heltec #1:  5V (J2 pin 2) + GND (J2 pin 1)
                │
                └──► 3V3 (J2 pin 3/4) → BME280 + MPU6050 + GPS (ver §3.4)
```

**Presupuesto orientativo** (medir en banco, no confiar en la hoja de datos):

| Consumidor | Típico | Pico |
|---|---|---|
| Pi Zero W v1 (idle, sin cámara) | ~150 mA | — |
| Pi Zero W + cámara + WiFi activo | ~250 mA | **~400-500 mA** (arranque) |
| Heltec V3 (RX) | ~30-50 mA | — |
| Heltec V3 (TX LoRa 21 dBm) | — | ~120 mA |
| BME280 + GPS + IMU | ~50 mA | ~80 mA (GPS fix) |

Con el step-up (≈85-90 % eficiente) el riel de 5V debería bancar **1 A con
margen**. Un 18650 de 3.4 Ah da varias horas de operación.

**Cuidados de diseño:**
- El pico de la Pi al arrancar es lo que hace caer el riel → poné **al menos
  470 µF electrolítico + 100 nF** junto al pin 5V de la Pi. Un brownout durante
  la escritura de la microSD la corrompe.
- Si alimentás la Pi por el **pin 5V del GPIO** (en vez del micro-USB), estás
  **salteando el fusible y la protección de polaridad** del conector: agregá en
  la PCB un fusible rearmable (polyfuse ~1.5 A) y un diodo de protección
  inversa, o usá un conector micro-USB en la placa.
- La Heltec **puede cargar su propia batería** por el conector JST 1.25; si la
  alimentás por 5V del J2, no uses las dos fuentes a la vez.
- **Carga de la 18650**: el módulo **XY-J02** (USB → batería) se encarga de
  cargar; verificar que la versión traiga **protección** (DW01 + dual MOSFET);
  si no, usar una 18650 con PCB integrada. Con eso, el **LM317** del DPD queda
  redundante (ahorra 4 g).
- **Verificar el step-up en banco**: que sostenga ≥4.8 V con la Pi arrancando
  (pico ~0.5 A) y la Heltec transmitiendo. Los módulos baratos de 1 A se caen.

## 3. UART: el detalle que importa

### 3.1 Pinout confirmado

**Raspberry Pi Zero W** (header de 40 pines):

| Función | GPIO | Pin físico |
|---|---|---|
| TX (sale de la Pi) | GPIO14 / UART TXD | **pin 8** |
| RX (entra a la Pi) | GPIO15 / UART RXD | **pin 10** |
| GND | — | pin 6 (o 9, 14, 20, 25, 30, 34, 39) |
| 5V | — | pin 2 o 4 |

**Heltec WiFi LoRa 32 V3** (header J2, según datasheet oficial Rev1.1/V3.2):

| Función | GPIO | Pin del J2 |
|---|---|---|
| TX | GPIO43 (U0TXD) | **pin 6** |
| RX | GPIO44 (U0RXD) | **pin 5** |
| GND | — | pin 1 |
| 5V | — | pin 2 |

### 3.2 ⚠ Los pines del J2 están compartidos con el chip USB (CP2102)

El datasheet dice explícitamente: *"RX: GPIO44, U0RXD, **connected to CP2102
TXD**"* y *"TX: GPIO43, connected to CP2102 RXD"*. O sea: el mismo UART0 que
usarías para hablar con la Pi está puenteado al conversor USB de la placa.

Consecuencias:
- Si dejás el USB conectado mientras la Pi transmite, **dos salidas pelean el
  mismo cable** (CP2102 y Pi) → datos corruptos o corriente entre chips.
- En vuelo el USB está desconectado, así que en la práctica *suele* funcionar,
  pero es una fragilidad que no querés en la placa definitiva.

**Recomendación para la PCB: usar una UART secundaria de la Heltec.**

El ESP32-S3 tiene 3 UARTs. Usá UART1 en **GPIO47/48** (header **J2**, pines 13
y 14), que están libres y no son de strapping:

| Señal | Heltec | Raspberry Pi |
|---|---|---|
| Heltec TX1 | GPIO47 (J2 pin 13) | GPIO15 / pin 10 |
| Heltec RX1 | GPIO48 (J2 pin 14) | GPIO14 / pin 8 |
| GND | J2 pin 1 (o J3 pin 1) | pin 6 |

En el firmware de la Heltec:
```cpp
// UART1 en pines libres; UART0 queda para USB/flash/debug
Serial1.begin(115200, SERIAL_8N1, /*rx=*/48, /*tx=*/47);
// y el protocolo LB135 se escribe/lee por Serial1
```

Ventajas: el USB sigue libre para programar y ver logs **sin desconectar nada**,
y no hay contención con el CP2102.

> Cualquier par de GPIOs sirve (evitá los "strapping" GPIO0/3/45/46 y los que
> usa la OLED: 17/18/21). GPIO47/48 son seguros y están en el header.

### 3.3 Reglas eléctricas del enlace

- **3.3 V en ambos lados** → conexión directa, **sin level shifter**.
- **Nunca** metas 5V a un pin UART (quemás el SoC).
- **GND común obligatorio** (por la PCB ya lo tenés; si usás cable, va el
  tercer hilo).
- **TX ↔ RX cruzados**: TX de la Pi al RX de la Heltec y viceversa. En la PCB
  nombrá los nets `PI_TX→HELTEC_RX` y `HELTEC_TX→PI_RX` para no invertirlos al
  rutear (es el error #1).
- **Resistencias serie de 100-330 Ω** en las dos líneas TX (no en RX): limitan
  corriente si hay un corto o un conflicto de drivers. A 115200 no afectan.
- 115200 8N1 (el default del protocolo LB135, `cansat/protocol.py`).

### 3.4 Sensores en la Heltec (BME280 + MPU6050 + GPS)

Los tres sensores van a la **Heltec #1** (computadora de vuelo), como pide el
DPD. La Pi **no** los lee directo: recibe sus datos por UART dentro del
protocolo LB135.

| Señal | Heltec | Sensor | Notas |
|---|---|---|---|
| I2C SDA | GPIO41 (J3 pin 8) | BME280 + MPU6050 | bus compartido: BME280 0x76/0x77, MPU6050 0x68 |
| I2C SCL | GPIO42 (J3 pin 7) | BME280 + MPU6050 | 3.3 V; los pull-ups de los módulos alcanzan para pistas cortas |
| UART2 RX | GPIO6 (J3 pin 17) | GPS TX | 9600 8N1 (default ATGM336H) |
| UART2 TX | GPIO5 (J3 pin 16) | GPS RX | |
| 3V3 | J2 pin 3/4 o J3 pin 2/3 | VCC de los 3 | GPS ~25 mA, BME ~1 mA, MPU ~4 mA |
| GND | J2/J3 pin 1 | GND de los 3 | |

- GPIO41/42 son JTAG por defecto: alcanza con no usar JTAG. Si se prefiere
  dejarlo libre, cualquier GPIO de la lista segura sirve (1, 2, 4, 5, 6, 7,
  19, 20, 47, 48) — el firmware se adapta con la matriz de I/O del ESP32-S3.
- **No usar**: GPIO0/3/45/46 (strapping), 17/18/21 (OLED), 33–38 (flash), 26.
- El GPS lleva **antena cerámica**: mirando "al cielo", sin cobre ni antena
  LoRa cerca (el chasis PETG no bloquea RF).
- MPU6050 rígido a la placa, alineado con los ejes del CanSat (documentar la
  orientación para el firmware).

### 3.5 Estación terrena: la segunda Heltec

- **No necesita PCB**: la Heltec #2 va por **USB-C a la notebook** (alimenta y
  da datos), con su antena IPEX conectada. El software de estación ya lee ese
  puerto serie.
- Si se quiere portátil, un power bank USB alcanza; no hace falta placa.

## 4. Checklist de diseño de la PCB

**Ruteo**
- [ ] 5V y GND: pista de **0.8 mm** (o plano de cobre) — la Pi sola pide picos de ~0.5 A.
- [ ] Señales UART: 0.25 mm, **cortas** y lejos del feed de antena LoRa y del
      flex CSI de la cámara (el CSI es ruidoso).
- [ ] Plano de GND continuo debajo de la Pi; evitá islas.
- [ ] **Keep-out de cobre** en la zona de la antena LoRa (IPEX + antena): el
      datasheet de la Heltec lo pide; una antena sobre cobre no irradia.

**Protección y depuración**
- [ ] Polyfuse + protección de polaridad en el riel de 5V.
- [ ] 470 µF + 100 nF junto al pin 5V de la Pi; 10 µF + 100 nF junto a la Heltec.
- [ ] **Test points** (o un header de 4 pines: 5V, GND, PI_RX, PI_TX) para
      conectar un adaptador USB-TTL y depurar el UART sin desarmar nada.
- [ ] LED de "power" en el riel de 5V (te salva en la campaña).

**Mecánica / vuelo**
- [ ] La Pi puede ir **montada sobre el header de 40 pines** soldado a la PCB
      (todo el conexionado pasa por ahí: no hay cables). Es lo más robusto.
- [ ] Si usás cables, que sean cortos, con conector JST con traba y **alivio de
      tensión**; la vibración del ascenso afloja todo lo que sea dupont.
- [ ] Sujetá la microSD (una gota de silicona o cinta kapton): se sale con
      vibración y perdés la misión.
- [ ] La cámara CSI: el flex es frágil y el conector tiene traba; fijalo.
- [ ] GPS mirando al cielo y lejos de la antena LoRa; keep-out de cobre
      también debajo de la antena cerámica del GPS.

## 5. Software del lado de la Pi (ya implementado)

```bash
# 1) Habilitar la UART de hardware (deshabilita la consola serie y el BT por UART)
sudo raspi-config   # Interface Options → Serial Port: consola NO, hardware SÍ

# 2) Escuchar a la Heltec y dejar el estado para el pipeline
python uart_listener.py --port /dev/serial0 --state outputs/uart_state.json

# 3) El pipeline usa ese estado: altitud/presión/temperatura + GPS por frame
python mission_pipeline.py --camera --uart-state outputs/uart_state.json ...
```

Prueba de banco sin volar (simulador → listener):
```bash
python sim_uart.py --frames 20 --interval 0.5 | python uart_listener.py --stdin
```

Prueba real: en la Heltec, mandar un paquete de prueba por `Serial1` cada
segundo y verificar en la Pi que el listener imprime `[1] pkt=... alt=... gps=...`.
Si no llega nada: cruzá TX/RX (error #1), revisá GND común (error #2) y que el
baudios sea 115200.

## 6. Referencias

- Datasheet Heltec WiFi LoRa 32 V3 (pinout J2/J3): `resource.heltec.cn`
- Protocolo LB135 (contrato de datos): `cansat/protocol.py`
- Guía de la Pi (paso a paso en la placa): `pi/guia_pi.md`
- Alineación con el DPD (qué dato alimenta qué requisito): `docs/ALINEACION-DPD.md`
