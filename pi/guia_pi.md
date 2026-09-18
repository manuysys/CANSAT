# Puesta a punto — Raspberry Pi Zero W v1 + AI Camera (IMX500)

> **Esta guía reemplaza a la anterior (que asumía Pi Zero 2 W).** La Zero W v1
> es ARM11/ARMv6 a 1 GHz, 32 bits, 512 MB: **no existen wheels de onnxruntime
> ni de PyTorch**, así que el pipeline corre con el backend de respaldo
> `cv2.dnn` (ver `cansat/onnxio.py`) y **sin detección YOLO en la CPU**
> (`--no-detect`; para detección está el NPU del AI Camera, que infiere
> on-sensor).
>
> Lo que se gana: la inferencia del IMX500 no usa la CPU. Lo que se pierde:
> velocidad de la red de terreno (medir en el paso 6 y ajustar `--frames`).

## 0. Materiales

- Raspberry Pi Zero W v1 + fuente micro-USB 5 V 2.5 A
- microSD 16–32 GB clase 10 + lector
- AI Camera + cable CSI **"Mini to Standard"** (viene en la caja): extremo
  mini de 22 pines a la Pi Zero, extremo estándar de 15 pines a la cámara
- PC con [Raspberry Pi Imager](https://www.raspberrypi.com/software/)

## 1. Flashear la microSD (en la PC)

1. Imager → **Choose Device**: Raspberry Pi Zero W.
2. **Choose OS**: *Raspberry Pi OS (32-bit)* — **NO 64-bit** (ARMv6 no puede).
3. **Choose Storage**: la microSD.
4. Botón de opciones (engranaje):
   - hostname: `cansat`
   - usuario: `pi` + contraseña del equipo
   - WiFi: SSID y clave — **solo 2.4 GHz** (la Zero v1 no ve redes de 5 GHz)
   - país WiFi: AR
   - **Enable SSH** (con contraseña)
   - locale teclado: `us` o `es`
5. Write. Insertar la SD, conectar la cámara **con la Pi apagada**, alimentar.

El primer arranque tarda 2–5 minutos (la v1 es lenta). Esperar a que el LED
verde deje de parpadear.

## 2. Primer contacto (desde la PC)

```bash
ssh pi@cansat.local
# si cansat.local no resuelve en Windows, buscar la IP en el router:
#   ssh pi@192.168.x.x
```

Verificar la placa **antes de seguir**:

```bash
uname -m        # debe decir: armv6l   (si dice aarch64/armv7l, NO es una Zero W v1)
python3 --version
```

## 3. Sistema y cámara IMX500

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y imx500-all      # firmware IMX500 + modelos preempaquetados
sudo reboot
```

Tras el reinicio (la primera carga de firmware de la cámara puede tardar
varios minutos):

```bash
# foto simple (en Lite no hay preview; se guarda a archivo)
rpicam-still -o ~/test.jpg -t 3000
ls -la ~/test.jpg                    # debe existir y pesar MB

# inferencia on-sensor con una red preempaquetada
ls /usr/share/rpicam-assets/ | grep -i imx500     # nombres exactos en tu versión
rpicam-still -o ~/det.jpg -t 5000 \
  --post-process-file /usr/share/rpicam-assets/imx500_object_detection.json
```

Si `rpicam-still` no detecta la cámara: revisar el cable CSI (contactos al
revés es el error más común), `dmesg | grep -i imx500` y actualizar firmware.

## 4. Dependencias del pipeline (SIN onnxruntime, SIN torch)

```bash
sudo apt install -y python3-venv python3-picamera2 python3-opencv
cd ~
# copiar el proyecto (ver paso 5), luego:
cd ~/cansat_seg_poc
# --system-site-packages es OBLIGATORIO: así el venv ve picamera2 y cv2 de apt
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install pyserial PyYAML
python -c "import cv2, numpy, serial, yaml; print('deps OK', cv2.__version__)"
```

También podés correr `bash pi/instalar_en_pi.sh`, que hace lo mismo y avisa si
detecta otra arquitectura.

## 5. Modelos y código (en la PC, una vez)

En la PC (donde está este repo), ejecutar:

```powershell
powershell -ExecutionPolicy Bypass -File pi\preparar_deploy.ps1
```

Eso arma `dist_pi/` con: el paquete `cansat/`, los scripts de vuelo, los
**modelos autocontenidos** (sin `.onnx.data`: OpenCV DNN no los lee) y el
baseline. Después copiar a la Pi (ajustar IP):

```powershell
scp -r dist_pi\* pi@cansat.local:/home/pi/cansat_seg_poc/
```

## 6. Prueba de humo y medición (la más importante)

Usar unas pocas imágenes (tiles de LoveDA o cualquier JPG/PNG):

```bash
cd ~/cansat_seg_poc && source venv/bin/activate
time python mission_pipeline.py --folder ~/tiles --frames 3 --interval 0 \
     --no-detect --no-damage --overwrite
```

Anotar **segundos por frame** (el `time` total dividido 3):

| s/frame del terreno | Qué hacer |
|---|---|
| ≤ 3 s | agregar `--no-damage` fuera y probar los modelos de daño por separado |
| 3–10 s | volar solo terreno (`--no-damage`); usar `--sampler` para priorizar |
| > 10 s | volar solo terreno con `--sampler`; la alternativa es el NPU del IMX500 |

El término medio es el que se espera en ARMv6. La misión del DPD tiene ~1–2
minutos de descenso: con 8 s/frame son ~10–15 frames, suficiente para el
corredor y el muestreo adaptativo.

### Detección de personas con el AI Camera (sin torch)

En la Zero v1 no hay ultralytics, pero el **NPU del IMX500 corre la red de
detección on-sensor** (costo de CPU casi nulo). Primero probarla sola:

```bash
python -m cansat.imx500 --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk --seconds 10
```

Si imprime detecciones, el pipeline la usa con:

```bash
python mission_pipeline.py --camera --det-backend imx500 --frames 5 --no-damage
```

⚠ Ese modo está implementado contra la API oficial de picamera2 pero **no fue
validado en hardware todavía**: es lo primero a probar en la Pi.

## 7. Vuelo

```bash
# --frames 1000 = grabar hasta que se corte la energía; ajustar tras medir
python mission_pipeline.py --camera --frames 1000 --interval 0 \
    --no-detect --no-damage --enhance \
    --det-backend imx500 \
    --pop-density 1500 \
    --out-dir /home/pi/vuelos/$(date +%Y%m%d_%H%M%S)
```

- `--det-backend imx500` reemplaza a YOLO por el NPU del AI Camera (si la
  cámara no es la AI Camera, usar `--no-detect`).
- `--enhance` ayuda con la nitidez (denoise + unsharp) y cuesta poco.
- `--p0-alt <altitud del predio>` para altitud absoluta del BMP280.
- Sin BMP280: simula atmósfera o usa `--uart-state` (ver paso 8); con
  `--uart-state` además **toma la posición GPS de la ESP32** y la asocia a cada
  frame (el DPD lo pide).
- `--pop-density` alimenta la **estimación de pérdidas humanas**
  (`cansat/casualties.py`): usar la densidad real del predio.
- La telemetría queda en `telemetry_<timestamp>.csv` + copia a `telemetry.csv`
  (y el `.jsonl` con la incertidumbre y los tiempos por etapa).

Arranque automático: `crontab -e` →

```
@reboot /home/pi/cansat_seg_poc/pi/run_flight.sh
```

## 8. UART con la ESP32 (contrato LB135)

En la Pi: `sudo raspi-config` → Interface Options → Serial Port:
consola de login **NO**, hardware UART **SÍ**. GPIO15 (RXD) ← TX de la ESP32,
GND común, 115200 8N1.

Prueba **sin hardware** (simulador local, mide gaps de radio):

```bash
python sim_uart.py --frames 12 --interval 0.05 | \
python uart_listener.py --stdin --out /tmp/uart_test.jsonl
```

Prueba real:

```bash
python uart_listener.py --port /dev/serial0          # deja outputs/uart_state.json
# y en otra terminal, el pipeline puede leer la última lectura de la ESP32:
python mission_pipeline.py --uart-state outputs/uart_state.json ...
```

## 9. AI Camera: inferencia on-sensor (integrada al pipeline)

El IMX500 corre la detección sin cargar la CPU. El pipeline ya la integra con
`--det-backend imx500` (ver paso 6). Ejemplos oficiales para comparar:

```bash
git clone https://github.com/raspberrypi/picamera2
python3 picamera2/examples/imx500/imx500_object_detection_demo.py \
    --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
```

Para usar **nuestros** modelos en la NPU hay que convertirlos en la PC con
Sony Edge-MDT (MCT → `imxconv-pt` → `imx500-package` → `.rpk`). Requiere PC
Linux con ≥4 GB RAM y Python 3.11; la segmentación semántica no es un camino
soportado por Ultralytics, así que es un proyecto aparte
(ver `docs/DATASETS-Y-TECNICAS.md` §3.3). El flood specialist re-entrenado ya
**pasa la auditoría** `audit_imx500.py` (opset 17, autocontenido, 1 entrada):
es el primer candidato a probar en el NPU.

## 10. Problemas conocidos

| Síntoma | Causa / solución |
|---|---|
| `No matching distribution found for onnxruntime` | Normal en ARMv6: no instalar; el pipeline usa `cv2.dnn` |
| `OpenCV DNN no pudo cargar ... .data` | El modelo tiene pesos externos: regenerarlo con `python tools/onnx_inline.py` |
| `rpicam-still: command not found` | Falta `sudo apt install imx500-all` (trae rpicam-apps) |
| Cámara no detectada | Cable CSI mal asentado (probarlo en la otra cara), `dmesg \| grep imx500`, `sudo apt full-upgrade` |
| `cansat.local` no resuelve | Usar la IP; en Windows instalar Bonjour o revisar el router |
| WiFi no conecta | La Zero v1 solo ve 2.4 GHz; verificar país en Imager |
| Todo lento (SSH incluido) | Normal: ARM11. El `--frames` de vuelo debe salir de la medición del paso 6 |
