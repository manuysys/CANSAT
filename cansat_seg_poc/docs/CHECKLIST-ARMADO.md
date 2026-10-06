# Checklist de armado e integración del CanSat (banco → pre-vuelo)

Integración del hardware REAL: **Pi Zero W v1 + AI Camera IMX500 + Heltec V3 #1
(vuelo) + BME280 + MPU6050** (+ GPS si se conecta). Todo el software ya está
validado en la placa (s/frame, NPU, UART, salud, systemd); este checklist es
para que el armado físico no introduzca fallas nuevas.

> Base: `docs/ELECTRONICA-PCB.md` (pinout y reglas eléctricas) y
> `pi/guia_pi.md` (software de la Pi). Marcar cada casilla en la corrida real
> y anotar en la tabla del final.

---

## 0. Inventario y herramientas

- [ ] Pi Zero W v1 + microSD (con el deploy `dist_pi` ya instalado).
- [ ] AI Camera IMX500 + cable flex **mini 22 pines** (el de la cámara es de 15).
- [ ] Heltec WiFi LoRa 32 V3 #1 (vuelo) + antena IPEX conectada.
- [ ] Heltec WiFi LoRa 32 V3 #2 (estación) — por USB a la notebook.
- [ ] BME280 (I2C 0x76/0x77) + MPU6050 (I2C 0x68).
- [ ] GPS ATGM336H (opcional por firmware; hoy simulado y declarado).
- [ ] Batería 18650 (con protección) + módulo XY-J02 + step-up 5V.
- [ ] Dupont cortos o PCB; destornillador, multímetro, estaño si va soldado.
- [ ] Notebook con el repo + `tools/vivo_pi.py` (para la prueba en vivo).

## 1. Alimentación (medir en banco, no confiar en la hoja de datos)

- [ ] Step-up sostiene **≥4.8 V con la Pi arrancando** (pico ~0.5 A) y la
      Heltec transmitiendo: medir con multímetro en el riel de 5 V.
- [ ] Capacitor **470 µF + 100 nF** junto al pin 5V de la Pi (si no está en la
      PCB, agregarlo antes de volar).
- [ ] Polyfuse/protección de polaridad si se alimenta por el pin 5V del GPIO.
- [ ] No usar a la vez la carga USB del XY-J02 y la alimentación por J2.
- [ ] Autonomía estimada: 18650 de 3.4 Ah → varias horas con el perfil de vuelo
      (Pi ~250 mA + Heltec RX + sensores ~50 mA).
- [ ] Medir el consumo REAL en banco durante 5 min de vuelo simulado.

## 2. Mecánica y montaje

- [ ] Pi montada firme (ideal: sobre el header de 40 pines soldado a la PCB).
- [ ] **microSD asegurada** (silicona o kapton): se sale con la vibración.
- [ ] Flex CSI con la traba cerrada y fijado (es frágil).
- [ ] Cámara mirando **hacia abajo** (nadir), sin obstrucción del domo/chasis.
- [ ] **Foco** ajustado con la vista en vivo (`pi/camara_en_vivo.py`, :8080) o
      `rpicam-still --shutter 4000` en la luz del predio.
- [ ] Heltec con antena IPEX conectada y **keep-out de cobre** en la zona.
- [ ] GPS (si va) mirando al cielo, lejos de la antena LoRa.
- [ ] MPU6050 rígido y alineado con los ejes del CanSat (documentar orientación).
- [ ] BME280 lejos de fuentes de calor (no pegado al step-up).
- [ ] Balance/centro de masa razonable; nada que se mueva en el ascenso.

## 3. Conexionado UART (el detalle que importa)

Configuración **validada hoy** (firmware actual, UART0 de la Heltec):

- [ ] Heltec **TX** (J2 pin 6, GPIO43) → Pi **RX** (GPIO15, **pin físico 10**).
- [ ] Heltec **RX** (J2 pin 5, GPIO44) → Pi **TX** (GPIO14, pin 8) — opcional
      para escuchar, la Pi sólo recibe.
- [ ] **GND común** obligatorio (J2 pin 1 ↔ Pi pin 6). Sin esto no hay enlace.
- [ ] 3.3 V en ambos lados: **sin level shifter**; nunca 5 V a un pin UART.
- [ ] 115200 8N1.
- [ ] ⚠ Para la PCB definitiva: usar **UART1 en GPIO47/48** (J2 pines 13/14)
      y liberar UART0 del CP2102 (ver `ELECTRONICA-PCB.md` §3.2). Si se cambia,
      actualizar el firmware (`Serial1.begin(115200, SERIAL_8N1, 48, 47)`).

Prueba rápida (con el firmware ya flasheado):

- [ ] En la Pi: `python uart_listener.py --port /dev/serial0 --state /tmp/uart_state.json`
      y ver `[1] pkt=... alt=... hum=...` cada segundo.
- [ ] Si no llega nada: cruzar TX/RX (error #1), revisar GND común (error #2),
      baudios, y que la consola serie de la Pi esté deshabilitada.

## 4. Sensores en la Heltec (I2C)

- [ ] BME280 y MPU6050 a **SDA GPIO41 / SCL GPIO42** (J3 pines 8/7), 3.3 V.
- [ ] Al boot, el firmware imprime el **scan I2C**: deben aparecer **0x76** (o
      0x77) y **0x68**. Si no aparecen, revisar cableado (ya pasó una vez: un
      dupont invertido los dejó fuera sin dañarlos).
- [ ] Humedad real en el contrato (`hum_pct` > 0) y `temp_C` coherente.
- [ ] `vcgencmd get_throttled` en la Pi = `0x0` durante la corrida.

## 5. Pruebas de banco (con todo armado, sin volar)

- [ ] **Cámara**: `python -m cansat.imx500 --model models_rpk/yolo11n_pp.rpk --seconds 5`
      → ~6-7 fps y personas detectadas si hay alguien enfrente.
- [ ] **Vuelo completo**: `FRAMES=5 bash pi/run_flight.sh` → telemetría +
      `pi_health.json` en la carpeta del vuelo, sin errores en `vuelo.log`.
- [ ] **UART integrado**: el mismo vuelo con la Heltec emitiendo → `hum_pct`
      real en el CSV y `uart_log.jsonl` con paquetes.
- [ ] **Eventos MPU**: sacudir la placa 4 s y dejarla quieta 15 s →
      `# EVENTO DESPEGUE` / `# EVENTO ATERRIZAJE` en el log del listener.
- [ ] **systemd** (opcional en banco): `sudo systemctl enable cansat-vuelo` y
      probar con override `FRAMES=2` (ver `guia_pi.md` §7).
- [ ] **Pull en vivo**: `pi/servidor_vivo.py` + `tools/vivo_pi.py` → la estación
      muestra el vuelo en vivo.

## 6. Pre-vuelo (el día de la campaña)

- [ ] **`bash pi/preflight.sh`** en verde (modelos, cámara+NPU, UART,
      **batería** >3.6 V, SD, temperatura, reloj, systemd).
- [ ] `--p0-alt <altitud del predio>` calibrado (QNH local) en `run_flight.sh`.
- [ ] Foco re-ajustado con la luz del predio (la vista en vivo es la forma).
- [ ] Batería al 100 % y **medida** antes de subir (no confiar en el LED).
- [ ] Hora de la Pi sincronizada (NTP con el celu/hotspot o manual).
- [ ] `FRAMES` acorde al descenso medido (~3 s/frame ⇒ 20-40 frames por minuto).
- [ ] microSD con espacio libre (>2 GB) y asegurada.
- [ ] Checklist de la ESTACIÓN (simulacro) firmado y el plan B a mano
      (`docs/demo_respaldo.webm` en la notebook).

## 7. Recuperación y post-vuelo

- [ ] Al recuperar: NO apagar de golpe si se puede (cortar energía de todos
      modos es normal; la telemetría se flushea por frame).
- [ ] Ingesta de la SD en la PC: `python tools/ingest_sd.py <ruta-SD>` (copia el
      vuelo a `outputs/mission`).
- [ ] Post-vuelo: `python post_flight.py --frames outputs/mission/full_res --no-edsr`
      y revisar `summary.json` + `pi_health.json`.
- [ ] Comparar `pi_health.json` (temp/throttled) con lo medido en banco.

## 8. Registro de corridas

| Fecha | Operador | Etapa (banco/pre-vuelo) | Resultado | Notas |
|---|---|---|---|---|
| | | | | |
| | | | | |
