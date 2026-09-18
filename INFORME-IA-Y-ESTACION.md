# Informe completo — IA de vuelo y Estación Terrena (CanSat LB135)

> **Equipo 135 — E.E.S.T. N.º 4 "I Brigada Aérea", El Palomar.**
> Documento de evidencias: qué está hecho, con qué datos, cómo se entrenó, qué
> funcionó, qué **no** funcionó y por qué. Todos los números salen de
> `cansat_seg_poc/outputs/metrics/*.json`, `MODELS.yaml` o los logs de
> entrenamiento — ninguno de memoria.
>
> Última actualización: 2026-09-18.

---

## 0. Resumen ejecutivo

| Área | Estado |
|---|---|
| Segmentación de terreno (5 clases) | ✅ mIoU **0.522** (v2@320) · **0.500** (v2@224, el de vuelo) |
| Detección de daño (two-stage) | ✅ **0.735** IoU de dañado en el dominio de vuelo (UAV) · 0.472 cross-event satelital |
| Inundación (FloodNet) | ✅ IoU flood **0.489** a 224 px |
| Fuego/humo (extensión) | ✅ media **0.782** en test (fuego 0.791 · humo 0.774) |
| Severidad del daño (5 niveles) | ✅ **0.575** IoU de colapso sobre edificios (medido) — reemplaza el 0.3 fijo |
| Personas/vehículos | ✅ VisDrone en PC (254 vs 42 del COCO) · 🟡 NPU IMX500 sin validar en placa |
| Mejora de imágenes con IA | ✅ EDSR x2 post-vuelo + `--enhance` a bordo |
| Estimación de pérdidas humanas | ✅ modelo de exposición con supuestos declarados y banda |
| Estrés ambiental | ✅ USI/GVI + **bruma (dark channel)** + **humidex (sensores)** |
| Estación terrena | ✅ 5 vistas, contrato de 34 columnas, 41 aserciones E2E en verde |
| Verificación | ✅ **200 tests**, ruff, compileall, auditoría IMX500 de los 6 ONNX de vuelo |

---

## 1. Qué pide el DPD (misión secundaria)

> *"Crear un mapa del terreno a partir de las imágenes… mejoradas en su calidad
> con ayuda de IA. Reconocer los porcentajes de vegetación, personas, edificios,
> cuerpos de agua y otros elementos… estudiar el estrés ambiental generado por
> la contaminación en las ciudades. Asimismo, en caso de que ocurra un desastre
> natural… detectar con mayor rapidez los daños materiales ocasionados y, según
> su magnitud, estimar las posibles pérdidas humanas resultantes."*

Cada frase tiene su implementación y su evidencia en este informe:

| Frase del DPD | Implementación | Sección |
|---|---|---|
| Mapa del terreno | `corridor_map.py` + trayectoria GPS en la estación | §4.1, §9 |
| Imágenes mejoradas con IA | EDSR x2 (`enhance_image.py`) + `--enhance` | §4.7 |
| Porcentajes de cobertura | `cansat/indices.py` → CSV/JSONL/radio/estación | §4.1 |
| Personas | VisDrone (PC) + SSD on-sensor del IMX500 (vuelo) | §4.6 |
| Estrés ambiental por contaminación | USI/GVI + bruma por imagen + humidex de sensores | §8 |
| Daños materiales | Consenso de modelos de daño (2-3 votantes) | §4.2 |
| Pérdidas humanas "según su magnitud" | `cansat/casualties.py` + colapso medido por el modelo de severidad | §7 |

---

## 2. Arquitectura

```
CanSat (vuelo)                                  Estación terrena
┌──────────────────────────────┐               ┌────────────────────────────┐
│ Heltec WiFi LoRa 32 V3       │  LoRa 915 MHz │ Heltec #2 (USB a notebook) │
│ (ESP32-S3): sensores BME280, │──────────────►│ uart_listener / web_server │
│ MPU6050, GPS ATGM336H        │               │  · app React (5 vistas)    │
│  · misión primaria (P/T)     │               │  · telemetry.csv (34 col.) │
│  · UART 115200 → Pi          │               │  · summary.json + entrega/ │
└──────────────┬───────────────┘               └────────────────────────────┘
               │ UART (3 cables, 3.3 V)
┌──────────────▼───────────────┐
│ Raspberry Pi Zero W v1       │  microSD 16-32 GB
│ + AI Camera IMX500 (CSI)     │──────────────►  recuperación → tools/ingest_sd.py
│  · terreno (cv2.dnn, 224 px) │
│  · daño / flood / fuego      │  post-vuelo (PC):
│  · NPU: detección personas   │  SegFormer-B5 + EDSR + corredor + informe
└──────────────────────────────┘
```

**Decisión de hardware (desvío del DPD, justificado en `docs/SUSTITUCION-HW.md`)**:
Pi Zero W v1 en vez de Zero 2 W y AI Camera IMX500 en vez de Camera Module 3.
Motivos: disponibilidad, menor consumo/masa y el NPU del IMX500 cubre la
detección de personas que la Zero v1 no puede correr (ARMv6 no tiene PyTorch ni
onnxruntime: el backend de vuelo es `cv2.dnn`).

---

## 3. Datasets utilizados

| Dataset | Uso | Licencia | Tamaño usado | Por qué se eligió |
|---|---|---|---|---|
| **LoveDA** (remap 5 clases) | Terreno | CC BY 4.0 | 2 522 train / 1 669 val | Segmentación aérea 0.3 m con clases alineadas al DPD (vegetación/edificios/agua/suelo/otros) |
| **xBD / xView2** | Daño (referencia satelital) | CC BY-NC-SA 4.0 | 2 283 tiles | Estándar de daño edilicio con 4 niveles; split **por desastre** |
| **RescueNet** | Daño (dominio de vuelo) + severidad | Investigación | 3 595 train / 449 val (UAV 3000×4000) | UAV post-huracán, GSD ≈0.05–0.08 m/px — el régimen real del CanSat (7.7 cm/px a 250 m) |
| **FloodNet** | Inundación | Sin licencia declarada (verificar) | 2 343 UAV | Único set UAV con clases flood/agua/edificio/ruta |
| **fire-smoke-seg** (FLAME vía Roboflow) | Fuego/humo (extensión) | CC BY 4.0 | 141/40/20 (256²) | Polígonos reales de fuego/humo desde UAV |
| **VisDrone** | Personas/vehículos | Challenge | 6 471 train | Detección aérea; fine-tune `yolov8n_visdrone.pt` |
| **BRIGHT** | ❌ descartado | CC BY / NC | 3.1 GB descargados | **Verificado: es pre-óptico + post-SAR.** El benchmark no tiene post óptico → inútil para el modelo de vuelo (§6) |

### 3.1 Por qué importa el dominio (lección central)

Los modelos de daño entrenados en **satélite** (xBD, 0.3 m/px) y en **UAV**
(RescueNet, ~0.07 m/px) rinden muy distinto en cada dominio. Medición directa
con el mismo protocolo (IoU de dañado sobre edificios):

| Modelo | xBD held-out (satélite) | RescueNet val (UAV) |
|---|---|---|
| xBD-only | **0.472** | 0.497 |
| Mezcla 82 % RescueNet | 0.004 | 0.686 |
| **Balanceado xBD×4 + 4k RN (vuela)** | 0.096 | **0.735** |

Como el vuelo es UAV, se eligió el balanceado. El xBD-only queda como
referencia cross-event del informe.

---

## 4. Modelos de IA

### 4.1 Terreno — `cansat_seg_terrain_v2.onnx` / `_224.onnx` / `_tiny_224.onnx`

- **Arquitectura**: DeepLabV3+ con backbone MobileNetV2, output stride 8, 5 clases.
- **Entrenamiento**: `train_terrain_v2.py` (12 épocas, 320 px, augmentación de
  vuelo: down-up + motion blur + ruido, pesos de clase 1/√frecuencia, cosine LR).
- **Matriz de tamaño de entrada** (mismo modelo, sin re-entrenar, Val completo):

| Entrada | 320 | 256 | 224 | 192 | 160 |
|---|---|---|---|---|---|
| mIoU | **0.5220** | 0.4941 | 0.4718 | 0.4400 | 0.4040 |

- **Re-entrenados a 224** (F1 2026-09-18): MV2@224 → **0.4996** (el de vuelo:
  mitad de cómputo que 320) y MV3-S+LR-ASPP@224 → **0.4397** (1.08 M params,
  4.3 MB, candidato NPU; no alcanza el objetivo de 0.48).
- **Artefactos**: ONNX opset 17 autocontenidos, auditados IMX500 y cargando en
  `cv2.dnn` (`cansat/onnxio.py`).

### 4.2 Daño — consenso de 3 modelos

1. **Principal** `cansat_damage3_mobilenetv2.onnx` (xBD, 3 clases):
   IoU dañado 0.107 (crops) / 0.215 (tile) en desastres nunca vistos.
   **Limitación medida**: en UAV satura (mediana 62.7 % de "daño" en tiles sin
   daño) → en vuelo no discrimina.
2. **Two-stage de vuelo** `cansat_damage_v3_bal.onnx` (xBD×4 + 4k RescueNet,
   loss de RescueNet): **0.735 IoU de dañado en el dominio UAV** (recall de
   frame 0.887 con umbral 10.2 %). Se enmascara con los edificios del terreno.
3. **Siamés** (pre/post) `cansat_siamese_damage.onnx`: 0.601 / 1.23 % de falso
   daño con pares idénticos. **Desactivado por defecto** (`--siamese-onnx`).

**Consenso** (`cansat/indices.py::diagnose`): votos por modelo con **umbrales
calibrados por modelo** (principal 10 %, two-stage 10.2 %), mayoría simple y
red de seguridad para daño fuerte.

### 4.3 Inundación — `cansat_flood_specialist_224.onnx`

- DeepLabV3+MV2, 3 clases (other/flood/agua normal), remapeo **corregido**
  contra el `class_mapping.csv` oficial.
- IoU flood **0.489** a 224 px (vs 0.483 a 320) con **2.04× menos cómputo**.
- Primer modelo del repo que pasó la auditoría IMX500.

### 4.4 Fuego/humo (extensión, no la exige el DPD) — `cansat_fire_smoke.onnx`

- DeepLabV3+MV2 (encoder inicializado desde el terreno), 256 px, 3 clases.
- Datos: 141/40/20 pares con polígonos de FLAME.
- **Test**: fuego **0.791**, humo **0.774** (media 0.782) — 2× el objetivo del plan.
- **No alucina**: fuego 0.00 % en LoveDA/RescueNet; humo ≤15 % en rural.
- Alertas: fuego >1 %, humo >30 % → `INCENDIO` / `HUMO EXTENSO`.

### 4.5 Severidad del daño (5 niveles) — `cansat_severity.onnx`

- `train_severity.py` reconstruye el recorte desde la máscara **original** de
  RescueNet (el nombre del tile guarda `y0`/`x0`), evitando regenerar 17 560
  tiles. Clases: other/intacto/menor/mayor/destruido.
- **IoU de colapso sobre edificios (mayor+destruido): 0.575** en el val de
  RescueNet (2 093 tiles); por clase: intacto 0.55, menor 0.38, mayor 0.30,
  destruido 0.48. ONNX auditado y cargando en `cv2.dnn`.
- En vuelo: `colapso_pct` medido por frame → `casualties.estimar(...,
  collapse_frac_medido=...)` reemplaza el supuesto 0.3. La telemetría registra
  `colapso_pct` + `colapso_fuente` ("medido"/"supuesto"). Esto es el
  *"según su magnitud"* del DPD.
- Entrenamiento cortado en la época 4 por tiempo (~10 min/época por decodificar
  las imágenes originales 3000×4000); queda margen para más épocas.

### 4.6 Personas y vehículos

- **PC/post-vuelo**: `yolov8n_visdrone.pt` (fine-tune de VisDrone): 254 personas
  vs 42 del COCO en imagen aérea de prueba.
- **Vuelo**: NPU del IMX500 (`--det-backend imx500`, `cansat/imx500.py`) con el
  `.rpk` SSD stock. 🟡 Pendiente validar en hardware a la GSD del descenso
  (a 250 m una persona ≈ 6 px: hay que medir cuántas detecta realmente).

### 4.7 Mejora de imágenes con IA

- **Post-vuelo**: EDSR x2 (`enhance_image.py`) sobre las capturas recuperadas.
- **A bordo**: `--enhance` (bilateral + unsharp) — barato en ARMv6.
- **Bench de degradaciones** (`bench_degradados.py`): matriz degradación ×
  mejorador con acuerdo de segmentación y nitidez.

---

## 5. Técnicas de entrenamiento aplicadas (y por qué)

| Técnica | Dónde | Resultado |
|---|---|---|
| **Split por desastre** (no por fila) | daño (todos los scripts) | Eliminó la fuga geográfica: el "0.32" viejo era memorización; el honesto es 0.10–0.47 |
| **Loss de RescueNet** (BCE localización + CE selectiva en foreground) | two-stage | Permitió entrenar con supervisión sólo en edificios |
| **Adaptación de dominio** (mezcla balanceada xBD×4 + RescueNet) | two-stage | UAV 0.497 → **0.735** |
| **Umbrales calibrados por modelo** | consenso | two-stage 10.2 % (F1 0.828, recall 0.887) en el dominio UAV |
| **Augmentación de vuelo** (blur de movimiento, ruido, down-up) | terreno | Robustece contra el movimiento del descenso |
| **Pesos de clase 1/√frecuencia** | todos | Clases raras (daño, flood, fuego) con señal |
| **Destilación (KD)** | terreno | Empata con el v2 (0.5196 vs 0.5219) sin costo extra |
| **Selección por métrica de misión** | daño | Mejor época por IoU de dañado **sobre edificios**, no mIoU dominado por fondo |
| **Dark channel prior** | estrés ambiental | Bruma/aerosoles sin modelo extra (clásico, barato) |
| **Humidex** (Magnus) | estrés ambiental | Combina T y humedad de los sensores como pide el DPD |
| **Auditoría automática IMX500** | todos los ONNX | `audit_imx500.py` en CI/pre-commit |

---

## 6. Técnicas que NO funcionaron (y por qué) — incluye MAE y pseudo-labels

Esta sección responde una pregunta concreta del equipo: *"la IA anterior intentó
MAE y pseudo-labels y no funcionaron; ¿por qué no las mencionaste?"*. **Sí se
conocían** — están en `legacy/` con su diagnóstico escrito — y por eso se
recomendaron alternativas en vez de repetirlas sin corregir la causa.

| Técnica | Qué pasó | Causa raíz | Estado |
|---|---|---|---|
| **MAE (autoencoder enmascarado)** | No mejoró y quedó abandonada | `legacy/mae_pretrain.py` entrenaba sobre `loveda_raw` **completo, incluido Val** (fuga), guardaba el encoder en la raíz y usaba stride 32 contra un modelo OS8. Además: con 2 522 imágenes etiquetadas y techo de LoveDA (~52 %), el pre-entreno auto-supervisado sobre el **mismo** dataset aporta poco frente a ImageNet | ❌ no se repite; el camino es más/better data y dominio, no self-supervision local |
| **Pseudo-labels** | La línea v3/v4 quedó rota | El teacher ensemble usaba el **flood roto** (máscaras de agua desplazadas), BN congelado, etiquetas desalineadas 1/3 y exportaba la última época | ⏸️ regenerables con el teacher corregido (terreno v2 + flood + daño UAV); baja prioridad |
| **BRIGHT cross-event** | Descartado tras descargar | **Es pre-óptico + post-SAR**: el post del benchmark es SAR (gris con speckle), no sirve para el modelo óptico de vuelo | ❌ documentado en `DATASETS-Y-TECNICAS.md` |
| **Siamés pre/post** | Desactivado por defecto | Re-entrenado bajó el falso daño a 1.23 %, pero el "pre" satelital nadir contra frames oblicuos es otro dominio; ensucia el consenso | 🟡 opcional (`--siamese-onnx`) |
| **INT8 dinámico** | Descartado | 23–26× más lento en ORT, no carga en `cv2.dnn`, acuerdo 90.5 % < 95 % | ❌ vuela FP32 |
| **CBAM (atención)** | Peor | 50.34 % vs 52.19 % del v2, +17 k params | ❌ |
| **MV3-S tiny** | No alcanza el objetivo | 0.4397 vs 0.4996 del MV2@224; su valor es el tamaño (4.3 MB → NPU) | 🟡 candidato NPU |
| **Principal re-entrenado con mezcla UAV** | Peor en ambos dominios | xBD 0.173 → 0.047 y RescueNet 0.476 | ❌ descartado |
| **xBD entrenado, evaluado en UAV** | Satura | 62.7 % de "daño" en tiles sin daño | ❌ por eso se adaptó el dominio |
| **QDQ + MCT + Edge-MDT** | Pendiente (no falló) | Requiere PC Linux + converter Sony | ⏸️ F4 |

**Conclusión honesta**: las técnicas que fallaron no eran malas en sí — estaban
mal implementadas o mal aplicadas al dominio. La receta que sí movió la aguja
fue **datos del dominio correcto (UAV) + split honesto + calibración medida**.

---

## 7. Estimación de pérdidas humanas (`cansat/casualties.py`)

Modelo de **exposición** explícito y auditable (no una predicción de víctimas):

```
personas_afectadas  = densidad × área_relevada × ocupación × fracción_dañada
pérdidas_estimadas  = afectadas × fracción_de_colapso × letalidad
```

- Supuestos por defecto (todos configurables y registrados en el JSONL):
  1 500 hab/km², ocupación 0.6, colapso 0.3, letalidad 0.1.
- Se publica con banda **[×0.5, ×2]** porque los tres últimos factores son
  incertidumbre pura.
- El **área** sale de la huella en tierra por frame (`ground_area_m2`, FOV real
  de la cámara) y la **fracción dañada** del consenso.
- ✅ El colapso es **medido** por `cansat_severity.onnx` (IoU 0.575 sobre
  edificios); si el modelo falta, cae al supuesto 0.3 y la telemetría lo
  declara (`colapso_fuente`).
---

## 8. Estrés ambiental (2026-09-18)

Tres capas que se combinan en `stress_idx` 0–100:

1. **Urbana** (existente): USI (edificios/vegetación), GVI, densidad, riesgo hídrico.
2. **Bruma/contaminación** (nueva, `cansat/stress.py`): *dark channel prior*
   (He et al. 2009) → `haze_pct` (% de frame con bruma densa), `visibility`,
   `atm_light`. Categorías: AIRE LIMPIO / BRUMA LEVE / MODERADA / DENSA (SMOG).
3. **Calor** (nueva): **humidex** = T + 0.5555·(e−10) con temperatura y humedad
   del BME280/UART → CONFORTABLE / DISCONFORT / ESTRÉS TÉRMICO / PELIGROSO.

`stress_idx = 50 % bruma + 25 % calor + 25 % carga antrópica`. Todo viaja en la
telemetría (CSV 34 columnas + JSONL con `contam`/`heat`/`visibility`).

---

## 9. Estación terrena (`EstacionTerrena_MuestreoDeDatos/`)

- **Servidor** `web_server.py` (stdlib, SSE): sirve la app, el contrato de
  telemetría (34 columnas), `/api/samples`, `/api/events`, degradación elegante
  sin CSV.
- **App React** (Vite + Three): vistas **Vuelo / Post / Informe / Presentación /
  Jurado**, corredor del terreno, trayectoria GPS, panel de Muestreo, detalle
  por frame con pestañas de imágenes (vis/ens_seg/enhanced/high_res), KPIs de
  daño, afectados y pérdidas estimadas, badges de diagnóstico.
- **Datos**: `telemetry.csv` (34 col.) + `telemetry.jsonl` (incertidumbre,
  tiempos por etapa, nodata, supuestos de casualties, contam/heat) +
  `summary.json` (schema v2) + `entrega/` (corredor, mejoras EDSR, evidencias).
- **Simulacro y demo**: `tools/simulacro.py`, `tools/make_demo_mission.py`
  (mundo sintético con zonas de incendio, inundación y sismo), `CHECKLIST-SIMULACRO.md`.
- **Verificación**: `npm run smoke` → **41 aserciones E2E en verde** (build de
  producción servido por Python, cero errores de consola).

---

## 10. Verificación y trazabilidad

| Qué | Resultado |
|---|---|
| Tests de vuelo (`pytest`) | **200 pasan** |
| Lint (`ruff check .`) | verde |
| Compilación (`compileall`) | verde |
| Auditoría IMX500 (`audit_imx500.py`) | los 6 ONNX de vuelo pasan (opset 17, autocontenidos) |
| Carga en `cv2.dnn` | verificada para cada ONNX de vuelo |
| Smoke de la estación | 41 aserciones E2E verdes |
| Registro de modelos | `MODELS.yaml` con métrica, fuente, hash y estado por artefacto |

---

## 11. Evidencias (imágenes generadas)

Las imágenes están **generadas y versionadas**; insertarlas en el informe final
es copiar y pegar desde estas rutas (relativas a la raíz del repo):

### IA de vuelo (`cansat_seg_poc/docs/evidencia/`)

| Archivo | Qué muestra |
|---|---|
| `01_terreno_gt_vs_pred.png` | imagen \| ground truth \| predicción del terreno (v2@320) |
| `02_dano_gt_vs_pred.png` | tile UAV de RescueNet \| GT de daño \| predicción del two-stage de vuelo |
| `03_flood_gt_vs_pred.png` | FloodNet \| GT \| predicción del flood 224 |
| `04_fuego_humo_gt_vs_pred.png` | FLAME \| GT fuego/humo \| predicción |
| `05_bruma_metrica.png` | imagen clara vs bruma sintética con las métricas del dark channel |

Otras evidencias ya existentes en `cansat_seg_poc/`:
`outputs/corridor_map.jpg` (corredor), `outputs/baseline.png` (siames),
`outputs/demo_damage.jpg`, `outputs/degradados_compare.jpg`,
`outputs/tta_compare.jpg`, `entrega/enhanced/` (EDSR), `entrega/ens_seg/`,
`entrega/b5_seg/` (SegFormer-B5), `dataset/visualizations/`.

### Estación terrena (`EstacionTerrena_MuestreoDeDatos/docs/evidencia/`)

| Archivo | Qué muestra |
|---|---|
| `01_ops_hero.png` | vista Vuelo con fase, cinta y bitácora |
| `02_scrubber_perfil.png` | perfil de altitud con scrubber |
| `03_bitacora.png` | bitácora de eventos |
| `04_replay.png` | replay de la misión |
| `05_modo_sol.png` | tema claro |
| `06_jurado_portada.png` | modo Jurado (portada) |
| `07_jurado_criticos.png` | modo Jurado (frames críticos) |
| `08_post_toast.png` | vista Post con notificación |

Para regenerarlas: `python tools/generate_evidence.py` (IA) y
`node tools/shots-r4.mjs` con el servidor arriba (estación).

---

## 12. Cómo reproducir

```bash
# Vuelo (PC)
cd cansat_seg_poc
pip install -r requirements.txt
python -m pytest tests -q                                  # 200 tests
python mission_pipeline.py --folder tiles --frames 3 --no-detect

# Re-entrenar (ejemplos)
python train_terrain_tiny.py --arch lraspp-mv3s --size 224 --epochs 30
python train_damage_v3.py --loss rescue --xbd-repeat 4 \
    --extra-manifest dataset/rescuenet_tiles/manifest_train_sub4000.csv
python train_flood_specialist.py --size 224
python train_fire_smoke.py --epochs 30
python train_severity.py --epochs 10

# Calibrar umbrales en el dominio de vuelo (GPU)
python tools/calibrate_thresholds.py --rescuenet --tiles 600 --model all --torch

# Deploy a la Pi
powershell -ExecutionPolicy Bypass -File pi/preparar_deploy.ps1 -Todos
scp -r dist_pi\* pi@cansat.local:/home/pi/cansat_seg_poc/

# Estación
cd ../EstacionTerrena_MuestreoDeDatos
python tools/make_demo_mission.py && python web_server.py
npm run smoke
```

---

## 13. Pendientes y riesgos (honestos)

| Pendiente | Impacto | Bloqueo |
|---|---|---|
| Validar en la Pi: s/frame, IMX500, personas | Alto (define el modo de vuelo) | microSD + AI Camera en mano |
| Conversión Edge-MDT (.rpk) de terreno/flood/fuego | Alto (NPU) | PC Linux con converter Sony (F4) |
| Severidad → colapso medido en casualties | ✅ hecho (0.575 IoU de colapso) | más épocas opcionales |
| Densidad poblacional real (WorldPop) por GPS | Medio | diseño pendiente |
| `model_id`/`quant` en la telemetría | Bajo-medio | pendiente |
| UI de la estación para bruma/calor/fuego | Bajo | los datos ya viajan |
| Pseudo-labels regenerados + re-destilado | Medio | baja prioridad (v3/v4 en legacy) |

**Riesgo principal**: la validación en hardware sigue pendiente; todos los
números de tiempo/cómputo en la Pi son estimaciones hasta medir con la placa.
