# Alineación con el DPD — CanSat "La Base" (CONAE 135)

Revisión del 2026-09-17 contra el **Documento Preliminar de Diseño** entregado.
Cada requisito del DPD → qué hay implementado → estado y qué falta.

> Leyenda: ✅ implementado y probado · 🟡 implementado, falta validar en la Pi ·
> ❌ falta.

---

## 1. Misión secundaria (el corazón del DPD)

> *"Crear un mapa del terreno a partir de las imágenes… mejoradas en su calidad
> con ayuda de IA. Reconocer los porcentajes de vegetación, personas, edificios,
> cuerpos de agua y otros elementos… estudiar el estrés ambiental… detectar
> daños materiales y estimar las posibles pérdidas humanas."*

| Requisito | Implementación | Estado |
|---|---|---|
| Segmentación semántica del terreno | `mission_pipeline.py` + `cansat_seg_terrain_v2.onnx` (DeepLabV3+ MobileNetV2, 5 clases, **mIoU 52.19 %** en Val completo) | ✅ |
| Porcentajes de vegetación/edificios/agua/suelo/otros | `cansat/indices.py` (fuente única) → CSV/JSONL/radio/estación | ✅ |
| **Personas** | Detección con el fine-tune de **VisDrone** en PC/post-vuelo (medido: 254 vs 42 personas del COCO en imagen aérea) y **NPU del AI Camera (IMX500)** on-sensor en vuelo (`cansat/imx500.py`, `--det-backend imx500`) | 🟡 el modo IMX500 necesita validarse en la Pi |
| Imágenes mejoradas con IA | EDSR x2 (`enhance_image.py`) en post-vuelo + `--enhance` (denoise+unsharp) a bordo | ✅ |
| Mapa del terreno | `corridor_map.py` (corredor apilado) + **trayectoria GPS** en la estación (`GpsTrack.tsx`) + overlays de segmentación | ✅ |
| Estrés ambiental por contaminación | USI (edificios/vegetación), GVI, densidad urbana, `flood_risk` y veredicto (`cansat/indices.py`) **+ bruma/aerosoles por imagen** (dark channel prior, `cansat/stress.py`) y **humidex** con temperatura+humedad del sensor; `haze_pct`/`humidex`/`stress_idx` (0-100) en la telemetría. Cubre el "junto con la información de los sensores" del DPD | ✅ |
| Detección de daños materiales | Consenso de 2-3 modelos (principal + two-stage adaptado a UAV + siamés opcional) con `cansat.indices.diagnose`; two-stage de vuelo = `cansat_damage_v3_bal.onnx` (F2 2026-09-18, IoU dañado 0.735 en el dominio UAV, umbral calibrado 10.2 %) | ✅ |
| **Estimación de pérdidas humanas** | `cansat/casualties.py`: modelo de exposición con supuestos declarados + banda; **fracción de colapso MEDIDA** por `cansat_severity.onnx` (F2b, IoU de colapso 0.575; si falta, cae al supuesto 0.3 y lo declara) | ✅ |

## 2. Asociación imagen ↔ telemetría (DPD: "hora, posición, altitud, presión, temperatura")

| Dato | Estado |
|---|---|
| Hora | ✅ `t_s` por frame |
| Presión/temperatura/altitud | ✅ BME280 (o BMP280) en la Pi, UART de la ESP32, o simulación |
| **Humedad (BME280)** | ✅ Protocolo v2 campo 23 (`hum_pct`), `uart_listener` la registra, el pipeline la asocia a cada frame y la estación la muestra en el detalle. **Cierra el gap del DPD**: se medía y no viajaba |
| **Posición GPS** | ✅ Protocolo v2 extendido con `lat`/`lon`, `uart_listener` los registra y `mission_pipeline --uart-state` los asocia a cada frame; la estación dibuja la trayectoria |

## 3. Hardware del DPD vs realidad

| DPD | Realidad del equipo | Acción |
|---|---|---|
| Raspberry Pi **Zero 2 W** | **Pi Zero W v1** (ARMv6) | El pipeline corre con `cv2.dnn` (sin onnxruntime) y `--no-detect`; el AI Camera cubre la detección. **Recomendación firme: conseguir una Zero 2 W** para volar el pipeline completo (daño incluido) |
| Camera Module 3 | **AI Camera (IMX500)** | El NPU reemplaza a YOLO en la Pi; ver `pi/guia_pi.md` |
| ESP32-S3 computadora de vuelo | Igual | El protocolo LB135 v2 + GPS cubre el enlace |
| Descenso 2–4 m/s, eyección ~250 m | Igual | `--apogee`/`--speed` y `sim_uart` reproducen el perfil |

## 4. Pruebas del DPD (las 8 de integridad)

| Prueba | Cómo se cubre |
|---|---|
| 1. Encendido y consumo | Fuera del alcance del software; el pipeline mide `ms_total` por frame (JSONL) |
| 2. ESP32 + Pi simultáneas | `uart_listener` + `mission_pipeline --uart-state` desacoplados por archivo de estado |
| 3. Sensores en conjunto | `sim_uart` inyecta el perfil atmosférico; el pipeline degrada a `sim_atmo` si falta el sensor |
| 4. Enlace a distancia real | `uart_listener` cuenta paquetes perdidos por número de paquete y checksums |
| 5. Cámara + IA en conjunto | `mission_pipeline --camera` (picamera2 o IMX500) |
| 6. **Simulación de descenso en banco** | `EstacionTerrena/tools/simulacro.py` + `CHECKLIST-SIMULACRO.md` (12 verificaciones) |
| 7. Verificación mecánica | Fuera del software |
| 8. Registro de resultados | `CHECKLIST-SIMULACRO.md` (tabla de corridas) + `docs/reporte_pruebas.md` generado |

## 5. Brechas abiertas (priorizadas)

1. **Generalización de los modelos de daño**: RESUELTO PARCIAL (F2
   2026-09-18). El two-stage de vuelo se re-entrenó con RescueNet (UAV) y
   sube a **0.735 de IoU de dañado en el dominio de vuelo** (era 0.497);
   a cambio pierde el dominio satelital (xBD 0.472 → 0.096), que queda como
   referencia cross-event. El principal xBD satura en UAV (62.7 % de "daño" en
   tiles sin daño). Umbral de voto recalibrado (10.2 %, F1 0.828, recall
   0.887). Detalle en `docs/DATASETS-Y-TECNICAS.md` y `MODELS.yaml`.
2. **Validar el modo IMX500 y el pipeline en la Pi** (medir s/frame reales).
3. **Umbrales del consenso sin calibrar** con frames reales (`--damage-threshold`
   ya es configurable; el JSONL registra todo para calibrar).
4. **Regenerar los pseudo-labels** de `train_v3` (los actuales usaron el flood
   roto como teacher de agua).
5. **Migrar los splits de daño a "por desastre"** en los 4 scripts de
   entrenamiento (el evaluador ya lo hace).
6. **Personas en vuelo**: validar cuántas detecta realmente el SSD del IMX500 a
   la GSD del descenso (7.7 cm/px a 250 m ⇒ una persona ≈ 6 px).

## 6. Extensión (no exigida por el DPD)

- **Fuego/humo** (F3 2026-09-18): modelo `cansat_fire_smoke.onnx` (3 clases,
  256 px, media IoU 0.782 en test) + alertas INCENDIO/HUMO EXTENSO en
  `cansat/indices.py` y columnas `fire_pct`/`smoke_pct` en la telemetría.
  El DPD no lo pide: se reporta como valor agregado.
