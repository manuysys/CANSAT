# Sustitución de hardware respecto del DPD — CanSat "La Base" (135)

El DPD presentado especifica **Raspberry Pi Zero 2 W** y **Camera Module 3**.
El equipo va a volar con **Raspberry Pi Zero W v1** y **Raspberry Pi AI Camera
(IMX500)**. Este documento justifica el desvío por escrito, como pide la
revisión, y demuestra equivalencia funcional con evidencia medida.

## 1. Qué cambia y qué no

| DPD | Vuelo real | Naturaleza del cambio |
|---|---|---|
| Pi Zero 2 W (quad-core ARMv8, 11 g) | **Pi Zero W v1** (ARM11/ARMv6, 9 g) | Placa disponible en el equipo; menor consumo |
| Camera Module 3 (12 MP, CSI) | **AI Camera IMX500** (12.3 MP, CSI + NPU) | Cámara disponible; agrega cómputo on-sensor |
| YOLO en la Pi (CPU) | **Detección en el NPU del IMX500** | El NPU hace lo que la Zero v1 no puede (no hay PyTorch en ARMv6) |
| onnxruntime | **OpenCV DNN (`cv2.dnn`)** | No existen wheels de onnxruntime para ARMv6 |

**No cambia**: la misión (mapa del terreno, porcentajes de cobertura, estrés
ambiental, detección de daño y estimación de pérdidas), el enlace con la
ESP32-S3 (UART 115200, protocolo LB135), la telemetría por LoRa, el paquete de
entrega (`telemetry.csv` + `summary.json` + evidencias + corredor) ni la
estación terrena.

## 2. Por qué (motivos técnicos y de proyecto)

1. **Disponibilidad**: la Zero W v1 ya está en el equipo; la Zero 2 W del DPD
   está pedida y aún no llegó. El software se adaptó para que **el mismo
   pipeline corra en las dos** (si la 2W llega a tiempo, se cambia la placa y
   no el código).
2. **Consumo y masa**: la Zero W v1 consume menos en reposo y pesa 2 g menos;
   ayuda al presupuesto de 223 g y a la autonomía con la 18650 + step-up de 1 A.
3. **La cámara elegida es *mejor* para este problema**: el IMX500 tiene un NPU
   integrado que corre la red de detección **en el sensor**. En la Zero v1 eso
   no es una optimización, es la única forma de tener detección de personas a
   bordo (ARMv6 no tiene PyTorch ni onnxruntime).
4. **El DPD no exige IA en tiempo real a bordo**: la Secuencia de Operaciones
   dice que la Pi *"captura, asocia y almacena para su posterior procesamiento"*
   y que las imágenes *"serán mejoradas y segmentadas una vez recuperado el
   CanSat"*. El flujo post-vuelo (SegFormer-B5 + EDSR + estación terrena) es el
   que cumple la misión ante el jurado, y corre en la PC sin restricciones.

## 3. Equivalencia funcional (con evidencia)

| Función | Cómo se cubre | Evidencia |
|---|---|---|
| Segmentación del terreno (5 clases) | `cansat_seg_terrain_v2.onnx` vía `cv2.dnn` en la Pi | mIoU **0.5219** (Val completo, 1669 imgs) |
| Detección de personas/vehículos | NPU IMX500 (`--det-backend imx500`) o VisDrone en post-vuelo | VisDrone: **254 personas** vs 42 del COCO en imagen aérea |
| Mejora de imágenes con IA | EDSR x2 en post-vuelo (offline, ya validado) | `enhance_image.py` + `entrega/enhanced/` |
| Telemetría imagen ↔ sensores | UART LB135 v2 con altitud/presión/temperatura/**GPS** | `uart_listener.py` + `mission_pipeline --uart-state` |
| Estimación de pérdidas humanas | `cansat/casualties.py` (exposición con supuestos) | Implementado y testeado (9 tests) |
| Modelos autocontenidos | `tools/onnx_inline.py` (cv2.dnn no lee `.onnx.data`) | `audit_imx500.py` en verde |

## 4. Lo que falta demostrar (medición en la placa)

Pendiente de la llegada de la microSD/AI Camera:

- [ ] **Segundos por frame** en la Zero W v1 con el modelo de terreno a 320 px
      (`time python mission_pipeline.py --folder tiles --frames 3 --no-detect --no-damage`).
      Estimación: 5–15 s/frame; la misión tolera 1 frame cada 5–10 s.
- [ ] **Detecciones del IMX500** a la GSD del descenso
      (7.7 cm/px a 250 m → una persona ≈ 6 px): `python -m cansat.imx500 --seconds 10`.
- [ ] **Consumo real** del conjunto con batería definitiva (medir mA en banco).

Si la Zero W v1 no alcanza el ritmo necesario, el plan B es **volar con la
Zero 2 W** (mismo código, 64-bit) y dejar la v1 como banco de pruebas.

## 5. Referencias

- `docs/ALINEACION-DPD.md` — cada requisito del DPD y su estado.
- `docs/ELECTRONICA-PCB.md` — conexionado y presupuesto eléctrico.
- `pi/guia_pi.md` — puesta a punto de la placa elegida.
- `MODELS.yaml` — qué modelo vuela, con métricas y hash.
