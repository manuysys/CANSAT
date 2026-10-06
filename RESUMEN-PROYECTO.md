# PROYECTO CANSAT LB135 — Inventario completo y trabajo realizado

> Documento consolidado (2026-09-21). Reúne **qué tiene el proyecto hoy** y
> **todo lo que se hizo**, con los números medidos. Los detalles finos están en
> `INFORME-IA-Y-ESTACION.md` (narrativa completa), `cansat_seg_poc/docs/`
> (decisiones, protocolos, reporte) y `MODELS.yaml` (registro de modelos).

---

## 0. Qué es

CanSat LB135 (E.E.S.T. N.º 4 El Palomar): una Pi con cámara IMX500 desciende
desde ~250 m, captura frames, los asocia a telemetría (batería, GPS, presión,
temperatura, humedad) y con IA produce mapas de daño/terreno, alertas y
estimación de personas afectadas. En tierra, una estación web muestra todo en
vivo y en post-vuelo.

**Dos aplicaciones en el mismo repo** (un solo git, remoto `manuysys/CANSAT`):

```
CANSAT/
├── cansat_seg_poc/                  # VUELO + POST-VUELO + IA (Python/torch/ONNX)
│   ├── cansat/                      # paquete compartido (21 módulos)
│   ├── tools/                       # herramientas (stress suite, consulta, evals…)
│   ├── tests/                       # 36 archivos de test (~400 tests)
│   ├── docs/                        # decisiones.yaml, reporte, protocolos, evidencia
│   ├── dataset/ · datasets/         # datos (no versionados)
│   ├── outputs/ · runs/ · weights/  # modelos y resultados (parcialmente versionados)
│   └── pi/                          # deploy a la Raspberry (guía, scripts, dist_pi)
├── EstacionTerrena_MuestreoDeDatos/ # ESTACIÓN WEB (Python stdlib + React/Vite)
│   ├── web_server.py                # servidor + API (sin dependencias)
│   ├── web-app/                     # app React (28 componentes)
│   ├── tools/                       # demo, simulacro, check-variantes, shots, smoke
│   └── CHECKLIST-SIMULACRO.md
├── .github/workflows/ci.yml         # CI activo (3 jobs)
├── INFORME-IA-Y-ESTACION.md         # informe narrativo (qué funcionó y qué no)
└── PLAN_MEJORA_IA_PI_ZERO_W_IMX500.md # plan de fases + auditoría V1–V8
```

---

## 1. Arquitectura en una línea

```
VUELO (Pi)                      POST-VUELO (PC)                ESTACIÓN (web)
cámara→segmentación        →   B5 + EDSR + ensemble      →    /api/mission (SSE)
→ daño/flood/fuego/sev     →   máscaras por frame        →    mapa + paneles
→ telemetría 34 col + JSONL→   consultas espaciales      →    Consulta Terrestre
→ alerta por radio (UART)  →   summary.json (contrato v3) →    vistas Post/Informe
```

El DPD no exige IA en tiempo real a bordo: el flujo post-vuelo en PC ya cumple
la misión secundaria; lo de a bordo es optimización.

---

## 2. Todo lo que tiene el proyecto

### 2.1 Modelos de vuelo (ONNX registrados en `MODELS.yaml`)

| Modelo | ONNX | Métrica medida | Estado |
|---|---|---|---|
| Terreno v2 (DeepLabV3+ MBv2 @320) | `cansat_seg_terrain_v2.onnx` | mIoU **0.5219** full Val (1669) | **VUELA** |
| Terreno v2 @224 | `cansat_seg_terrain_v2_224.onnx` | mIoU 0.4996, **2.04× menos cómputo** | candidato Pi |
| Terreno tiny @224 (MV3-S+LR-ASPP) | `cansat_seg_terrain_tiny_224.onnx` | mIoU 0.4397 | candidato NPU (no cumple en CPU) |
| Daño principal (xBD) | `cansat_damage3_mobilenetv2.onnx` | IoU edificios 0.107–0.215 según protocolo | VUELA |
| Daño two-stage de vuelo ep6 (xBD×4+RescueNet) | `cansat_damage_v3_bal_ep6.onnx` | **RescueNet val 0.780** · xBD held-out 0.414 · umbral voto 10.2 % (F1 0.832, recall 0.900) | **VUELA** |
| Siamés (cambio pre/post) | `cansat_siamese_damage.onnx` | IoU 0.601; alucinación pre==post 1.23 % | opcional, OFF por defecto |
| Flood specialist @224 | `cansat_flood_specialist_224.onnx` | IoU flood **0.489**, agua 0.502 | VUELA (opcional) |
| Fuego/humo | `cansat_fire_smoke.onnx` | IoU fuego 0.791–0.806, humo 0.718–0.774 | VUELA (opcional) |
| Severidad (colapso medido) | `cansat_severity.onnx` | IoU colapso **0.633**; intacto 0.584 / menor 0.385 / mayor 0.415 | VUELA (opcional) |
| Tipo de desastre (7 clases) | `cansat_disaster_type.onnx` | acc val 0.989; **LOEO 0.3681** (24 eventos); MIL 0.3277 (no supera) | opcional, política `fire_only_v1` |
| Detección personas (YOLO) | `models/yolov8n_visdrone.pt` | VisDrone 254 personas vs COCO 42 en la imagen de prueba; v2 mAP50-95 0.164 en val | VUELA en PC |

Además: exports `_int8` (dinámicos, **descartados**), `best_damage_v3.onnx` (xBD
de referencia), `distilled` (KD empató 0.5196 vs 0.5219).

### 2.2 Modelos post-vuelo (no vuelan)

| Modelo | ONNX | Métrica |
|---|---|---|
| SegFormer-B5 @512 | `cansat_seg_terrain_segformer_b5_512.onnx` | **mIoU 0.5664**, pixel-acc 0.7237 (Val completo 1669) |
| EDSR x2 (super-resolución) | `models/EDSR_x2.pb` | **287.9 s/frame 1024² en CPU** (solo post-vuelo) |
| Especialista de vías | `cansat_vias_floodnet.onnx` | **IoU 0.515** FloodNet val (gate ≥0.50 PASA); cross-domain LoveDA 0.169 |

### 2.3 Datasets usados (detalle y licencias en README/INFORME)

- **LoveDA** (Zenodo, CC BY 4.0): terreno 5 clases (Train 2522 / Val 1669 + Test).
- **xBD/xView2** (CC BY-NC-SA 4.0): daño estructural; Tier 3 → 5661 tiles, 19 eventos.
- **RescueNet** (UAV óptico): dominio de vuelo del two-stage; val 2093 tiles.
- **KATE-PD**, **CRASAR-U-DROIDs**: daño/severidad multi-desastre.
- **FloodNet Track 1**: inundación (398 máscaras crudas; remapeo oficial flood=1/agua=2).
- **FLAME/fire-smoke-seg** (CC BY 4.0): fuego/humo (train 141 / valid 40 / test 20).
- **VisDrone**: detector de personas/vehículos (mAP medido).
- **WorldPop**: densidad poblacional (grilla 0.1°, 15 467 celdas).
- **EarthVQA** (académico, no comercial): QA para el benchmark del parser (Val 57 202 = 51 distintas; gated en HF, QA local en `dataset/earthvqa/`).

### 2.4 Pipeline de misión (`mission_pipeline.py`)

- **Telemetría 34 columnas** + JSONL enriquecido (incertidumbre, tiempos, nodata,
  supuestos de casualties, `model_ids`+`quant`, tipo de desastre, **detecciones
  con `xyxy`**, **aviso OOD/drift** `ood_score`/`ood_flag`, contam/heat).
- Segmentación de terreno con **TTA**, **CRF-lite**, **suavizado temporal** y
  **máscara de nodata por borde**; sampler adaptativo (HIGH/MEDIUM/LOW + buckets
  high_res/full_res/thumb).
- **Consenso de daño** de 3 modelos con umbral 10 % y red de seguridad 25 %.
- Flood, fuego/humo, severidad (colapso medido) y tipo de desastre con política
  `fire_only_v1` (solo “incendio” confirmado; el resto abstiene).
- **Estrés ambiental**: USI/GVI/densidad + bruma por *dark channel prior* +
  humidex (BME280) → `stress_idx` 0-100.
- **Casualties**: exposición con supuestos declarados (densidad × área ×
  ocupación PAGER × daño × colapso × letalidad) + banda ×0.5–×2.
- **Alerta por radio** con protocolo v2 unificado (`cansat/protocol.py`).
- **Detección** por YOLO (PC) o NPU IMX500 (`--det-backend imx500`).
- **ONNX backend dual**: onnxruntime o `cv2.dnn` (Pi Zero v1 ARMv6).

### 2.5 Contrato de datos (estación ↔ vuelo)

- **`summary.json` schema v3**: alertas, pérdidas, veredictos, y el bucket nuevo
  `masks` (máscaras de consulta) además de vis/high_res/full_res/thumb/ens_seg/enhanced.
- **Máscaras por frame** (`entrega/masks/<src>_<fuente>.png`, PNG gris, 255=sin
  dato): terreno, daño, daño2, flood, fuego, severidad y vías.
- Compatibilidad hacia atrás en `normalize_summary` y degradación elegante si
  falta cualquier artefacto.

### 2.6 Consulta Terrestre (`cansat/consultas.py` + `tools/consulta.py`)

Motor **simbólico** (sin LLM): plantillas español/inglés → operaciones
deterministas sobre máscaras y telemetría.

- Operaciones: `area`, `count` con **buffer métrico**, `length_fraction` por
  esqueleto, `distance`, `exists` y **personas con posición** (conteo por zona,
  a ≤ r metros de un objeto y distancia mínima/media usando el **punto de apoyo
  del bbox**).
- Zona dibujada (polígono lon/lat) con **georreferenciado aproximado declarado**
  (FOV, norte arriba, sin heading).
- Si no mapea: **“consulta no soportada”** con sugerencias; nunca inventa.
- **Benchmark EarthVQA**: cobertura **41.4 %** (23 681/57 202; Basic
  Counting/Judging 85.7 %) y **precisión 100 %** de mapeo (15/15) y de rechazo
  (36/36) sobre el inventario completo de 51 preguntas, etiquetado a mano
  (`docs/benchmarks/`).

### 2.7 Estación terrena (`EstacionTerrena_MuestreoDeDatos/`)

- **Servidor stdlib** `web_server.py`: API GET (`/api/mission`, `/api/frame`,
  `/api/samples`, `/api/summary`, `/api/health`, `/api/consulta`,
  `/api/gradcam`, SSE `/api/events`), proxy `/img` (incluye `outputs/gradcam/`),
  cache por mtime, sin dependencias.
- **App React** (Vite): vistas **Vuelo / Post-vuelo / Informe / Presentación /
  Jurado**, 27 componentes: escena 3D del descenso con fallback SVG, corredor,
  mapa offline con tiles locales, timeline, bitácora, panel de muestreo,
  **panel de Consulta** con chips y zona dibujada, detalle por frame, KPIs,
  modo sol de campo, tour de introducción.
- **Contrato v3** en tipos (`types.ts`, `summarySchema.ts`) y validación
  (`validate.ts`).
- **Demo sintética** coherente (`tools/make_demo_mission.py`: mundo con
  inundación/incendio/sismo, 12 frames, máscaras y detecciones), **simulacro en
  vivo** (`tools/simulacro.py`: stream acelerado con dropout, ráfaga y trunco) y
  **checklist de simulacro** con 4 variantes, 3 de ellas automatizadas
  (`tools/check-variantes.mjs`).
- **Smoke E2E** `tools/smoke-v4.mjs`: **64 aserciones verdes**, cero errores de
  consola, sobre el build de producción servido por Python (incluye generación
  real del overlay Grad-CAM, rutas de error del endpoint y el bloque de
  confianza; los mensajes de consola llevan la URL para depurar).
- **Confianza limitada por estrés**: `summary.json` advierte frames bajo bruma
  densa/calor peligroso (umbrales de `cansat/stress.py`); sección en el Informe.
- **Grad-CAM integrado**: botón "explicar" en el detalle con selector de modelo
  (vuelo/xBD), overlay mostrar/ocultar y caché en `outputs/gradcam/`.
- **Evidencia**: capturas en `docs/evidencia/` (13 en la estación, 10 en vuelo).

### 2.8 Herramientas principales

| Herramienta | Para qué |
|---|---|
| `evaluate.py` | Eval unificada ONNX/PyTorch sobre LoveDA (JSON versionado + confusión) |
| `tools/stress_suite.py` + `cansat/corrupt.py` | Aptitud de vuelo: limpio vs 8 corrupciones (`sombras`/`vibracion` nuevas); el mismo módulo es augment de entreno (`AUG_UAV`) |
| `tools/consulta.py` / `bench_parser_earthvqa.py` | Consulta Terrestre y benchmark del parser |
| `tools/calibrate_per_class.py` | T por clase del tipo: experimentada y **rechazada** (ECE 0.131 vs 0.123) |
| `firmware/heltec_lb135_uart/` | Emisor LB135 v2 para Heltec V3 (PlatformIO); formato validado contra el parser |
| `tools/calibrate_per_class.py` | T por clase del tipo: experimentada y **rechazada** (ECE 0.131 vs 0.123) |
| `train_disaster_type.py --loss` | CE / balanceada / focal + LOEO en tramos (`--eventos`); balanceada **rechazada** (0.317 vs 0.330) |
| `train_*.py` (~15) | Entrenamientos: terreno, daño, flood, fuego, severidad, tipo, vías… |
| `prepare_*.py` / `download_*.py` | Preparación/descarga de datasets |
| `export_onnx.py` | Exportador **único** (opset 17, `dynamo=False`, auditoría) |
| `quantize_onnx.py` / `quantize_all.py` | QDQ estático (testeado) / dinámico (descartado) |
| `audit_imx500.py` | Auditoría de requisitos Sony (opset, autocontenido, tamaños) |
| `tools/onnx_inline.py` | Empotra `.onnx.data` para `cv2.dnn`/IMX500 |
| `generate_report.py` | Genera `docs/reporte_pruebas.md` desde `decisiones.yaml` + métricas |
| `tools/calibrate_thresholds.py` | Calibración de umbrales con frames reales |
| `tools/ingest_sd.py` | Ingesta de la microSD con verificación SHA-256 |
| `tools/build_ood_reference.py` | Referencia OOD desde LoveDA Val (clases + ExG + sombras) |
| `tools/select_tiles_for_annotation.py` | Cola de anotación (active learning: entropía + rareza + estrés) |
| `tools/gradcam.py` | Grad-CAM mínimo para auditar qué píxeles deciden una predicción |
| `cansat/calibracion.py` | ECE + temperature scaling + sets adaptativos (APS) del tipo |
| `tools/generate_evidence.py` | Figuras GT vs predicción para el informe |
| `pi/` | Guía de Pi, deploy, conversión IMX500, dist_pi |

### 2.9 Tests y CI

- **~400 tests locales**: métricas, protocolo, índices, casualties,
  conformal, sampler, CRF, nodata, xbd, summary (incluye confianza limitada),
  onnxio (mapeo por nombre y backends), imx500 (parser SSD), masks, consultas,
  corrupt (incluye `sombras`/`vibracion`), stress suite, post_flight, helpers
  del pipeline, enhance_image, quantize **QDQ end-to-end**, export (slow),
  `generate_report` y calibración por clase.
- En CI (`-m "not slow and not gpu and not dataset"`): **366 pasan** +
  8 deseleccionados.
- **CI en GitHub Actions activo y verde** (3 jobs): vuelo (ruff 0.16.8 fijado +
  compileall + tests + imports + generador del informe), deps de vuelo y build
  de la estación. Los errores se publican en **anotaciones del check**.
- `pre-commit` con ruff 0.16.8, chequeos de archivos y `generate_report --check`
  local.

### 2.10 Documentación y evidencia

- `INFORME-IA-Y-ESTACION.md`: informe narrativo completo (13 secciones).
- `cansat_seg_poc/docs/decisiones.yaml`: **fuente única** de decisiones y
  limitaciones; `generate_report.py` arma el reporte de ahí.
- `docs/reporte_pruebas.md`: reporte generado (números de `outputs/metrics/*.json`).
- `docs/PROTOCOLO-JOPLIN-NEPAL.md`, `ALINEACION-DPD.md`, `DATASETS-Y-TECNICAS.md`,
  `CONVERSION-IMX500.md`, `ELECTRONICA-PCB.md`, `SUSTITUCION-HW.md`.
- `docs/evidencia/` (vuelo) y `EstacionTerrena_.../docs/evidencia/` (estación).
- `docs/benchmarks/`: JSON del benchmark EarthVQA + muestra adjudicada a mano.

---

## 3. Números clave (una tabla)

| Qué | Resultado |
|---|---|
| Terreno de vuelo | mIoU 0.5219 (Val 1669) · @224 0.4996 |
| Terreno post-vuelo (B5) | mIoU 0.5664 |
| Daño two-stage de vuelo (ep6) | IoU UAV 0.780 · xBD held-out 0.414 |
| Daño xBD cross-evento | Joplin 0.233 (0.495 sobre edificios) · Nepal 0.127 (0.278) |
| Flood | IoU 0.489 |
| Fuego/humo | IoU fuego 0.791–0.806 · humo 0.718–0.774 |
| Severidad (colapso) | IoU 0.633 |
| Tipo de desastre | LOEO 0.3681 · MIL 0.3277 (no supera) · umbral incendio 0.99 (conformal FPR≤5 %, recall 0.291) |
| Vías (consulta) | IoU 0.515 FloodNet val · cross-domain 0.169 |
| Stress suite (limpio→peor) | terreno 51.0→16.3 (niebla) · daño UAV 0.403→0.198 (niebla) · flood 0.489→0.256 (lluvia) · fuego 0.801→0.306 (lluvia) · daño2 UAV: motion_blur −0.021, escala −0.026 (n=60) |
| Parser Consulta (EarthVQA) | cobertura 41.4 % · precisión 100 % (15/15 y 36/36) |
| Estación | smoke **64 aserciones verdes** · checklist 4/4 variantes · Grad-CAM + confianza integrados |
| Calibración por clase | ❌ ECE 0.131 vs 0.123 global + rompe ranking de incendio → no adoptada |
| Aug UAV (daño) | ❌ fine-tune limpio 0.374 → 0.265 → no adoptado (A/B correcto en V11) |
| Tipo rebalanceado | ❌ LOEO 0.317 vs 0.330 (banda ±0.04); redistribuye sin subir la media → no adoptado |
| Confianza limitada | ✅ bloque en `summary.json` + sección en el Informe (bruma ≥45 %, humidex ≥46) |
| **Pi Zero W v1 (medido 2026-09-25)** | **tiny@224 ~3.0 s/frame** (1.77 s segmentación) · v2@224 **243 s/frame** → vuelo CPU con tiny; resto post-vuelo |
| **AI Camera IMX500 (validada 2026-10-03)** | cámara OK en la Zero v1 · **persona real detectada** (demo oficial 77 %; pipeline `personas=1` con caja) · 2 bugs del módulo corregidos (lista de tensores + COCO-80) · captura: usar `--shutter` corto (60 ms automáticos → blur) |
| EDSR | 287.9 s por frame 1024² en CPU |
| CI | 3 jobs verdes · 366 tests en CI |

---

## 4. Lo que se hizo (61 commits, por etapas)

**Base (F0/estructura)**
- Repositorio reorganizado en `cansat_seg_poc/` + `EstacionTerrena_MuestreoDeDatos/`;
  shims y exports unificados en `export_onnx.py`; `evaluate.py` unificado (eliminó
  4 mIoU distintos hardcodeados); `generate_report.py` con `decisiones.yaml` como
  fuente; auditoría IMX500; README/plan de mejora.

**Terreno (F1)**
- v2 DeepLabV3+ MBv2 @320 (mIoU 0.5219) como modelo de vuelo; variante @224
  (0.4996, 2× menos cómputo); tiny 224 para NPU (0.4397); destilación (empató);
  matriz de tamaños de entrada; auditoría y pin de ONNX autocontenidos.

**Daño (F2)**
- Split **por desastre** (fin de la fuga geográfica); loss estilo RescueNet;
  two-stage de vuelo adaptado a UAV (0.497→0.735 en RescueNet); calibración de
  umbrales (10.2 % UAV); protocolo Joplin/Nepal (coincide con el paper: Nepal se
  derrumba); KATE-PD/CRASAR y siamés re-entrenado (opcional, OFF).

**Flood + fuego (F3)**
- Remapeo FloodNet corregido contra el `class_mapping` oficial + especialista
  224 (IoU 0.489); modelo de fuego/humo y alertas integradas; contrato creció a
  34 columnas.

**Severidad, estrés, tipo (F2b/F5+)**
- Severidad multi-desastre (colapso IoU 0.633) que reemplaza el supuesto 0.3;
  estrés ambiental (dark channel + humidex + ExG/sombras) en telemetría y UI;
  clasificador de tipo de desastre con LOEO honesto (0.3681), conformal para
  abstención de incendio (umbral 0.99) y política `fire_only_v1`; MIL probado y
  **descartado** por no superar al clasificador por tile.

**Post-vuelo**
- SegFormer-B5 (0.5664) + EDSR + ensemble + CRF + temporal + corredor; corrección
  del salto silencioso de la 2ª pasada; summary canónico; B5 medido y EDSR medido.

**Estación**
- Migración a React/Vite; mapa offline con tiles locales; contrato v3
  (tipo de desastre, fuente de colapso, `model_ids`); demo sintética coherente;
  simulacro con anomalías; checklist de simulacro; smoke E2E; pérdidas estilo
  PAGER con densidad WorldPop por GPS; modo sol/jurado/presentación.

**Tanda de propuestas externas (misma sesión, 2026-09-21)**
- Se evaluaron **23 propuestas** de optimización/investigación (V9/V10 en
  `PLAN_MEJORA`). Veredictos: 6 ya implementadas o redundantes (graph opt, CRF,
  pruning, TTA-BN, MoE, UDA), 2 descartadas por premisa equivocada (early-exit,
  DisasterKD), 2 de investigación condicionadas a datos/GPU (diffusion, continual),
  y las aceptadas se implementaron:
  - **OOD/drift**: `cansat/ood.py` + referencia de LoveDA Val + badge en la
    estación (avisa cuando la escena no se parece a la de entrenamiento).
  - **Calibración honesta**: `cansat/calibracion.py` (temperatura + ECE + APS
    con probs multiclase del LOEO).
  - **Perfiles de vuelo**: `--perfil rapido` (terreno@224 + detección + estrés,
    NO emite veredicto de daño) y `--perfil completo` (el validado).
  - **Active learning**: cola de anotación priorizada para el primer vuelo real.
  - **Grad-CAM**: CLI de explicabilidad para auditar predicciones raras.

**Tanda del plan externo (2026-09-21/22, `PLAN_MEJORA_IMPLEMENTACION_COMPLETA.md`)**
- La mitad ya existía (active learning, drift, tool Grad-CAM). Lo nuevo:
  - **Grad-CAM en la estación**: `/api/gradcam` + overlay + caché (evidencia
    `10_gradcam.png`, decisión `gradcam-estacion`); smoke 58 → **62**.
  - **Calibración por clase**: experimentada y **rechazada** (ECE 0.131 vs
    0.123; rompe el ranking de incendio; `calibracion_por_clase.json`).
  - **Augmentación UAV**: ops `sombras`/`vibracion` en `cansat/corrupt.py`
    (fuente única entreno+suite, 8 corrupciones) + `--aug-uav` en
    `train_damage_v3.py`; fine-tune **rechazado** (limpio 0.374 → 0.265;
    `aug_uav_dano.json`).
  - **UART Heltec**: firmware emisor LB135 v2 (`firmware/heltec_lb135_uart/`),
    formato validado contra el parser; pendiente flashear y probar.
  - Diferido a V11: MoE, UDA, difusión/LoRA, capas Sentinel Hub (rompen el modo
    offline), replay continuo.

**Validación de hardware final (2026-10-02/03)**
- **AI Camera IMX500**: cámara OK, **persona real detectada por el NPU** (demo
  oficial 77 %; pipeline `personas=1` con caja). Se corrigieron **2 bugs** de
  `cansat/imx500.py` que sólo aparecen en hardware: `get_outputs()` devuelve
  lista `[boxes, scores, classes]` y el SSD es COCO-80 (persona=0).
- **UART por GPIO15**: 40/40 paquetes Heltec→Pi y pipeline `--uart-state`.
- **Heltec + BME280 + MPU6050**: sensores reales en el contrato v2 (scan I2C
  0x76/0x68); los sensores sobrevivieron a una conexión invertida.
- **One-vs-rest (tipo)**: rechazado con evidencia (AP macro 0.286 vs 0.261).
- **Hallazgo de captura**: 60 ms de exposición automática → motion blur;
  integrado `--shutter`/`--gain` al pipeline (verificado en placa: 7995 µs,
  gain 16) y documentado para el vuelo.

**Bring-up Pi Zero W v1 (2026-09-25, hardware real)**
- SD 16 GB flasheada (Raspbian 13 **Trixie**, no Bookworm) → SSH por clave,
  `apt full-upgrade`, `imx500-all` instalado, escritorio apagado
  (`multi-user.target`; la imagen era Desktop), `throttled=0x0`.
- `dist_pi` copiado (400 MB) + `pi/instalar_en_pi.sh` (OpenCV 4.10 de apt,
  venv `--system-site-packages`, pyserial/PyYAML; sin onnxruntime en ARMv6).
- **Medición real**: tiny@224 = 1.77 s de segmentación / ~3.0 s/frame total;
  v2@224 = **243 s/frame**; los MV2 (daño/flood/fuego/severidad) ~4 min c/u →
  **decisión `modelo-vuelo-tiny`**: vuela el tiny, el resto post-vuelo en la PC
  (el DPD no exige IA en tiempo real a bordo). Evidencia:
  `docs/benchmarks/pi_zero_w_sframe.json`; guía y MODELS.yaml actualizados.
- Pendiente de hardware: AI Camera (IMX500) y UART real (ESP32-S3 disponible).

**Tanda A+B (2026-09-22): mejoras a números bajos**
- **A (tipo rebalanceado)**: `--loss balanceada/focal` + LOEO en tramos
  (`--eventos`); resultado 0.317 vs 0.330 → **rechazado** (el cuello es
  dominio, no desbalance; decisión `tipo-balanceado`).
- **B (confianza por estrés)**: bloque `confianza_limitada` en `summary.json`
  (aditivo, umbrales de `stress.py`) + sección en el Informe + espejo TS;
  smoke 62 → **64** (decisión `confianza-estres`).

**Tanda del handoff (sesión previa)**
- **MIL**: media por bolsa 0.3277 vs 0.3681 → documentado como técnica que no
  funcionó; queda el clasificador por tile.
- **Stress suite**: `cansat/corrupt.py` + `tools/stress_suite.py` con 5 tareas
  (terreno, daño xBD, two-stage UAV, flood, fuego) y 6 corrupciones; números
  publicados en `outputs/stress_suite.json` y en decisiones.
- **Especialista de vías**: `prepare_vias.py` + `train_road_specialist.py`;
  FloodNet-only IoU 0.515 PASA el gate (variante mixta 0.489 no pasaba;
  cross-domain 0.169 declarado); corre solo en post-vuelo.
- **Máscaras + Consulta Terrestre**: persistencia de máscaras por frame, motor
  simbólico, CLI, endpoint en la estación, panel con chips y polígonos,
  georreferenciado aproximado declarado, contrato v3 con bucket `masks`.
- **EarthVQA**: QA real descargado (gated), benchmark del parser con
  adjudicación manual completa (51 preguntas), dos falsos soportes corregidos
  (`uncultivated` y `construction land`), parser bilingüe, evidencia versionada.
- **Consulta v2 (personas)**: el pipeline persiste detecciones (`tipo+xyxy`) en
  el JSONL; el motor responde personas por zona, a ≤ r metros y distancia
  persona→objeto (punto de apoyo del bbox); puntos verdes en el mapa; smoke 56.
- **CI**: el workflow estaba en `cansat_seg_poc/.github/` (GitHub nunca lo
  corría) → movido a la raíz con 3 jobs; **verde**; ruff fijado a 0.16.8 y 20
  findings corregidos; shadcn vendorizado; informe portable y reproducible;
  errores visibles en anotaciones.
- **Higiene**: 9 shims eliminados; pseudo-labels/re-destilado cerrados como
  descartados; auditoría IMX500 documentada como manual; `simulacro.py`
  corregido (encoding UTF-8, rompía el server en Windows).
- **Tests**: 35 nuevos (post_flight, helpers del pipeline, enhance_image, QDQ
  estático end-to-end, export slow, cv2 5) → 330 locales.
- **Hallazgo real**: OpenCV **5.0.0 da logits absurdos en `cv2.dnn`** con el
  ONNX de vuelo (0.01 de acuerdo vs ORT; 4.13 >0.99) → pin `<5` en
  requirements, aviso en runtime y limitación `opencv-5-dnn`.

---

## 5. Lo que NO funcionó (honesto, documentado)

| Técnica | Resultado | Qué queda |
|---|---|---|
| MAE pre-entrenado | No transfería; quedó en `legacy/` | — |
| Pseudo-labels | Etiquetas desalineadas/BN congelado | Descartado formalmente |
| QAT eager / INT8 dinámico | No carga en cv2; 23-26× más lento | FP32; QDQ estático pendiente en placa |
| Destilación KD | Empató (0.5196 vs 0.5219) | No se persigue |
| MIL (bolsa=evento) | 0.3277 vs 0.3681 del tile | Clasificador por tile |
| Siamés con pre oblicuo | Alucinaba cambio | OFF por defecto |
| Mix FloodNet+LoveDA para vías | 0.489 (no pasaba el gate) | FloodNet-only 0.515 |
| Calibración por clase (tipo) | ECE 0.131 vs 0.123 global; rompe ranking de incendio | No adoptada (`calibracion_por_clase.json`) |
| Augmentación UAV (daño) | Fine-tune limpio 0.374 → 0.265 | No adoptado; A/B correcto en V11 (`aug_uav_dano.json`) |
| Tipo con pesos por clase | LOEO 0.317 vs 0.330 (ruido ±0.04); mejora 3 eventos, hunde 4 | No adoptado; checkpoint intacto (`tipo_balanceado.json`) |
| Un binario por tipo (one-vs-rest) | AP macro 0.286 (multiclase) vs 0.261 (OvR); weighted 0.329 vs 0.287 | No adoptado; el cuello es cross-evento, no las cabezas (`tipo_one_vs_rest.json`) |

---

## 6. Pendientes (solo hardware / placa)

1. ✅ **microSD y Pi operativa** (2026-09-25): Raspbian 13 Trixie, OpenCV 4.10,
   `imx500-all` instalado, SSH por clave, escritorio apagado.
2. **s/frame**: ✅ medido (tiny ~3.0 s, v2 243 s → `modelo-vuelo-tiny`).
   **AI Camera/IMX500**: ✅ validada a fondo (2026-10-04): cámara OK; detector
   de vuelo **YOLO11n** (~6.8 fps sin CPU, mAP 0.374 vs 0.218 del SSD; 3/3
   personas en la cadena real); segmentación DeepLabV3+ a ~1.1 fps; pose
   descartada con evidencia. 🟡 Pendiente: conversión de nuestros modelos a
   `.rpk` (Edge-MDT/Linux; se intenta vía Docker/WSL2).
   **UART real**: ✅ **cerrado**: Heltec V3 + BME280 + MPU6050 (21/21 en PC),
   ESP8266/CH340 (25/25), **GPIO15 en la Pi 40/40** y **soak de 10 min:
   597/597 paquetes, 0 pérdidas** (`soak_uart.json`); eventos de vuelo
   (despegue/aterrizaje) del MPU6050 validados en hardware
   (`eventos_vuelo_mpu.json`).
   **Operación**: ✅ salud de la Pi (`pi_health.json`, 40-42 °C sin
   throttling), autostart systemd (`pi/cansat-vuelo.service`) y pull en vivo
   Pi→estación (`pi/servidor_vivo.py` + `tools/vivo_pi.py`) validados.
   **LoRa (2026-10-05)**: enlace implementado en el firmware — TX de vuelo y RX
   de estación (SX1262 915 MHz, RadioLib); TX flasheado y OK, el enlace RF
   end-to-end queda pendiente de la segunda Heltec (`-e heltec_lora_rx`).
   **GPS (2026-10-05)**: soporte del ATGM336H en el firmware (UART2/NMEA);
   con fix usa lat/lon reales y sin fix declara el perfil simulado.
3. **Conversión Edge-MDT (.rpk)** de nuestros modelos: ✅ **HECHA (2026-10-05)**
   — cadena Docker/WSL2 con `edge-mdt[pt]` de PyPI (público, sin registro); el
   tiny de terreno corre en el NPU (`network.rpk`: memoria 4.28/8 MB = 54 %,
   KPI 4.5 ms, CPU libre). Pendiente: salida más chica para subir el end-to-end
   (0.4-1.2 fps) e integrarlo como opción del pipeline.
4. **INT8**: re-medición QDQ estático en la placa con `cv2.dnn` (2026-10-04)
   para decidir FP32 vs INT8 con número medido.
5. Checklist de simulacro: ✅ **firmado** (Juan Manuel Iglesias, 2026-10-03,
   variantes 1-4 + guion de 12 pasos en vivo).
6. **A/B de augmentación UAV (daño)**: en curso (2026-10-04, mismo init/receta
   ± aug con escala suave; resultado en `docs/benchmarks/aug_uav_dano_ab.json`).

---

## 7. Cómo correr lo principal

```bash
# Tests + lint (CI)
cd cansat_seg_poc && pytest -q -m "not slow and not gpu and not dataset" && ruff check .

# Evaluación del modelo de vuelo (Val completo, escribe outputs/metrics/)
python evaluate.py --onnx outputs/cansat_seg_terrain_v2.onnx

# Stress suite (5 tareas, 6 corrupciones)
python tools/stress_suite.py

# Stress del modelo de VUELO (tiny@224)
python tools/stress_suite.py --tasks terreno \
    --onnx-terreno outputs/cansat_seg_terrain_tiny_224.onnx --img-size 224

# Soak UART del contrato v2 (COM4 en la PC, /dev/serial0 en la Pi)
python tools/soak_uart.py --port COM4 --minutes 10

# Consulta Terrestre (CLI)
python tools/consulta.py --q "área de edificios inundados" \
    --masks entrega/masks --telemetry outputs/mission/telemetry.csv

# Informe (local; requiere outputs/ y entrega/)
python generate_report.py && python generate_report.py --check

# Estación (demo + smoke)
cd ../EstacionTerrena_MuestreoDeDatos
python tools/make_demo_mission.py
python web_server.py            # http://localhost:8000
node tools/smoke-v4.mjs         # 64 aserciones
node tools/check-variantes.mjs  # variantes del checklist
```
