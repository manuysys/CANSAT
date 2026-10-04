# Demo de presentación — "Pi en vivo" → estación (2026-10-03)

Cadena completa, probada end-to-end:

```
Heltec (BME280 real) --UART GPIO15--> Raspberry Pi (AI Camera + tiny@224
+ detección on-sensor IMX500) --scp--> PC (post-vuelo completo con ORT)
--> Estación terrena (mapa, consulta, informe)
```

## Preparación (una vez, antes de presentar)

1. **Heltec**: encendida por USB de la PC y cableada a la Pi:
   `TX → GPIO15 (pin 10)` y `GND → GND (pin 6)`. La Pi ya tiene el UART
   configurado (`/dev/serial0`, consola serie off).
2. **Pi**: encendida, misma red WiFi, IP fija **192.168.68.240**
   (`cansat.local` también resuelve). SSH por clave desde la PC.
3. **PC**: estación construida (`web-app/dist`) y repo de vuelo al día.

## Comando único (en la PC)

```powershell
cd EstacionTerrena_MuestreoDeDatos
powershell -ExecutionPolicy Bypass -File tools\demo_pi.ps1 -Frames 12
```

Hace todo: captura en vivo (12 frames ≈ 45 s), trae la misión, corre el
post-vuelo completo (sin EDSR, ~1-2 min) y copia misión + entrega a la
estación. Al terminar:

```powershell
python web_server.py        # → http://localhost:8000
```

## Qué mostrar

1. **Pi capturando** (opcional, en otra terminal):
   `ssh pi@192.168.68.240` y dejar corriendo
   `bash ~/cansat_seg_poc/pi/demo_vivo.sh 12` — se ven los frames, las
   detecciones de personas del NPU y el UART con los sensores reales.
2. **Estación** (vista Vuelo): corredor con los frames capturados, badge OOD,
   detecciones, y el detalle por frame.
3. **Post-vuelo**: daño/flood/fuego/severidad calculados en la PC + Informe
   (Imprimir/PDF) y póster (`docs/poster_LB135.png`).

## Plan B

- **Sin red**: la captura se puede hacer igual y pasar la misión por SD
  (`tools/ingest_sd.py`); la estación funciona 100% offline.
- **Sin Heltec**: `demo_vivo.sh` avisa y sigue con atmósfera simulada.
- **Sin cámara**: el pipeline degrada a `--folder` con tiles de prueba.
- **Sin tiempo**: `-Frames 6` (≈20 s) o `-SaltarPostVuelo` para mostrar solo
  la captura y el corredor.

## Detalles técnicos

- Captura: `mission_pipeline --camera --no-damage --onnx tiny --img-size 224
  --det-backend imx500 --shutter 8000 --gain 16 --uart-state …` (el daño va en
  la PC; `--shutter` evita el motion blur de la exposición automática).
- Los frames crudos se guardan en `full_res/` (modo cámara) para el
  post-vuelo de la PC.
- Tiempos medidos: captura ~3.5 s/frame; post-vuelo de 12 frames ~1-2 min en
  la PC (sin EDSR).
