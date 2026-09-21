# Plan ENORME de mejora IA — CanSat LB135 para Pi Zero W ARMv6 + IMX500
*Solo lectura, cero cambios. Integra lo que dejó la otra IA en cansat_seg_poc/docs/ + auditoría dura + búsqueda 2026-09-18.*

## 0. Punto de partida medido (no opiniones)

| Modelo actual | Métrica honesta | Fuente |
|---|---|---|
| Terreno DeepLabV3+MV2 5cl 320px FP32 51MB | mIoU **0.5219**, pix 0.6862 | `outputs/metrics/val_20260917_*.json`, `MODELS.yaml:48` |
| Daño principal 3cl | IoU dañado **0.104** val-por-desastre / 0.132 train | `eval_sliding.py --split val`, `MODELS.yaml:103` |
| Daño two-stage | **0.025** sin GT / 0.436 con edificios GT | `MODELS.yaml:130` |
| Siamés re-entrenado | IoU 0.601, falso 1.23% medio | `train_siamese_damage.py --neg-frac 0.3` |
| Flood specialist | IoU flood **0.483** | `MODELS.yaml:175` |
| Personas | **NO MEDIDO** | `MODELS.yaml:198` |

La otra IA ya dejó en `docs/DATASETS-Y-TECNICAS.md:20` la conclusión correcta: el terreno está bien, el eslabón débil es daño + personas. Este plan la extiende con constraint final que vos fijaste: **Zero W v1 ARMv6 512MB + IMX500**. Eso cambia todo.

## 1. Constraints duros que mandan el diseño

**ARMv6 (`pi/guia_pi.md`, `requirements-flight.txt`, `cansat/onnxio.py`):**

- Sin `onnxruntime/torch/ultralytics` (no hay wheels `armv6l`). Solo `cv2.dnn.readNetFromONNX` vía `apt python3-opencv` + `venv --system-site-packages`.
- `cv2.dnn` **no lee `.onnx.data` externa**, no expone `size_px/n_classes`, no soporta `DynamicQuantizeLinear/ConvInteger` (INT8 dinámico inútil). Solo FP32 o QDQ estático si los ops existen.
- ARMv6 = ARM11 sin NEON útil. OpenCV usa baseline ARMv7, Tengine deprecado. Esperar **~8s/frame a 320px** con DeepLabV3+MV2 51MB. Misión 1-2 min = 10-15 frames.

**IMX500 (`audit_imx500.py`, docs Sony Edge-MDT 2026):**

- Solo feed-forward estático, 1 entrada, sin `Loop/If/Scan/GridSample/RoiAlign/ScatterND/Einsum/NonZero`, opset **15-20** (vos usás 17, bien), salida `logits`, `<200MB` para converter pero **<8MB memoria total** (pesos + tensores) para correr on-sensor. PicoSAM2 2025 demuestra: 1.3M params / 1.22MB cuantizado / 14.3ms en IMX500 es el régimen que sí vuela.
- Flow obligatorio en PC Linux 4GB Python 3.11: `FP32 ONNX → Sony MCT (quantize+compress INT8) → imxconv-pt -i model_mct.onnx -o out/ → packerOut.zip → packager → .rpk → cansat/imx500.py`.
- Segmentación **no** está en lista Ultralytics soportada, pero converter acepta ONNX custom si los ops están cubiertos. Transformers (SegFormer-B5 338MB, 2539 nodos, LayerNorm/Softmax/Erf/GELU) **no pasan**.
- Estado hoy: solo `terrain_v2` + `flood` + base + `cbam` pasan `audit_imx500.py`. `damage3/damage_v3/distilled` son opset 18 + shell 0.3MB + `.data` huérfana → NO pasan. Siamés tiene 2 entradas → inconvertible directo. B5 338MB → imposible.

> Decisión arquitectónica que la otra IA sugirió en `DATASETS-Y-TECNICAS.md:87` y confirmo: **no cambiar de backbone grande antes de la competencia, pero sí achicar para IMX500**. El camino es doble: `cv2.dnn` ultraliviano como fallback + `.rpk` on-sensor como primario.

## 2. Estrategia general: 1 modelo que vuela + N que ayudan

```
PC (entrena todo) ──→ ONNX FP32 320px ──→ eval_sliding honesto
        │
        ├──→ rama A: cv2.dnn fallback (ARMv6): FP32 autocontenido <15MB, input 192-256px
        └──→ rama B: IMX500 on-sensor (ideal): MCT INT8 + imxconv-pt → .rpk <8MB
                    detección personas/vehículos YA va por .rpk SSD existente
```

Vuelo Zero W: `--no-damage` por defecto, `--det-backend imx500`, 1 modelo seg/detección por frame.
Post-vuelo PC: SegFormer-B5 + EDSR + ensemble (sin constraints).

No intentar correr 6 modelos/frame en Zero v1. `decisiones.yaml:performance-pi` ya midió que flood corría 2x por frame + baseline releído por frame.

## 3. Track T0 — Higiene bloqueante (1 día, hacer primero)

La otra IA lo listó en `MODELS.yaml:293` y sigue vigente:

1. Re-exportar `damage3/damage_v3/distilled` con `export_onnx.py --arch damage --opset 17 dynamo=False` + `tools/onnx_inline.py --out-dir dist_pi/outputs`. Hoy son 0.3MB + `.data` 51MB que `cv2.dnn` no carga.
2. Borrar al cerrar informe: `outputs/_roto_backup/` 106MB, `*_BACKUP.onnx`, `yolo11n.pt` raíz huérfano, `*_int8.onnx` dinámicos 13MB x4 (no cargan en cv2, 23-26x más lentos en ORT), `runs/` 37MB, `__pycache__`.
3. `pi/preparar_deploy.ps1` hoy solo empaqueta terreno+flood. Agregar flag `-Todos` ya existe pero `dist_pi/` en disco solo tiene 2/5 inline → regenerar.
4. `export_b5_640.py` roto (`best_terrain_b5.pth` inexistente + clase `SegformerTerrain` inexistente) → borrar o redirigir a `export_onnx.py --arch segformer`.
5. Licencias: FloodNet sin licencia, `EDSR_x2.pb` "ver repo origen", xBD CC BY-NC-SA (derivados no-comercial + sharealike). Bloqueante DPD si no se declara.

Criterio done: `audit_imx500.py --all` en verde para los que vuelan + `onnx_inline.py --check` OK + `mission_pipeline.py --folder 3f --no-detect --no-damage` <10s/f en Pi.

## 4. Track T1 — Terreno: de 51MB a <8MB sin perder 52% mIoU

No necesitas más mIoU, necesitas **10x menos costo**.

**4.1. Reducción input (mayor ROI, 0 re-entreno pesado):**
Matriz a medir con `evaluate.py` + `bench_degradados.py`: 320 → 256 → 224 → 192 → 160. Esperado: -1 a -3 pts mIoU por 2x menos FLOPs. Para DPD con GSD 7.7cm/px a 250m, 192px sigue resolviendo edificios. Entregable: `train_terrain_v2.py --size 224/192` + export + curva mIoU vs ms en Pi.

**4.2. Backbones IMX500-friendly (solo Conv+ReLU+BN+Depthwise, sin attention):**
Orden de prueba, todos exportables a ONNX opset 17 y convertibles MCT:

- `MobileNetV3-Small + LR-ASPP` (~2.5M, ~0.5G FLOPs): reemplazo directo de MV2, ~2x más rápido en ARM CPU. Prioridad 1.
- `STDC1 + PP-LiteSeg-T` (STDCNet + FLD/UAFM): 73-75% mIoU Cityscapes a 150-270 FPS GPU, solo convs. Prioridad 2.
- `FastSCNN / BiSeNetV2-STDC`: dos ramas detail/context, muy barato a baja resolución. Prioridad 3.
- `Depthwise U-Net estilo PicoSAM2` (1.3M, 336M MACs, 1.22MB INT8, 14.3ms IMX500): si STDC no entra en 8MB, este es el diseño que sí entra. Requiere entrenar desde cero con distilación SAM2 → solo si sobra tiempo.
- NO: PIDNet-S (7.6M), DDRNet, SegFormer, DINOv2 encoder en vuelo (solo para KD offline).

**4.3. Compresión:**
Pruning estructurado de canales 30-50% en ASPP/decoder + fine-tune 5 épocas → QDQ estático `quantize_onnx.py` (nunca ejecutado) → MCT PTQ → `imxconv-pt`. El INT8 dinámico actual se descarta. Validar con `validate_int8_mission.py` (umbrales 0.95/0.95/1.00).

**4.4. Destilación:**
`distill_terrain.py` ya existe (B5→MV2, KL T=4.0, alpha 0.5). Re-correr con pseudo-labels regenerados (hoy contaminados por flood roto) + teacher B5 512px. Ganancia típica 1-3 pts sin costo inferencia.

## 5. Track T2 — Daño: de IoU 0.10 a utilizable (máximo ROI misión)

**5.1. Fix metodológico (días 1-2, sin datos nuevos):**

- Split por desastre en los 4 scripts (`train_damage.py:77`, `v2:89`, `v3:84`, `siamese:226`). `eval_sliding.py` ya tiene `split_por_desastre`. El 0.32 viejo era fuga geográfica.
- Loss RescueNet en `train_damage_v3.py`: BCE localización edificio + CE selectiva solo foreground (ya a medio camino con `ignore_index=0` + `BCE(p(edif)=1-p(other))`). Paper RescueNet reporta orden de magnitud en clases intermedias.
- Patch-mining 80% centrado en edificio (ya en v3) + oversample colapso total (clase rara).
- Calibrar `--damage-threshold` (hoy 10% vs daños reales 0-2%) + `--uncert-max` con JSONL p95 del simulacro. Sin esto el consenso nunca dispara.

**5.2. Datos nuevos para más tipos de desastre (clave de tu pedido):**

| Dataset | Qué aporta | Licencia | Encaje concreto |
|---|---|---|---|
| **BRIGHT** Zenodo `14619797`, 4538 pares 1024px, 14 eventos óptico+SAR 0.3-1m, DFC 2025 Track II | Sismos, huracanes/ciclones, incendios, inundaciones, volcanes, explosiones. Baselines cross-event | Abierta (óptico guiado parcial) | **Usar solo subset óptico** (sin SAR, no lo tenés). Protocolo train-en-13, test-en-1-no-visto. Reemplaza/aumenta xBD |
| **RescueNet** 4494 UAV 3000x4000 post-Michael, 10 clases píxel + 3 niveles imagen | Edificio no-damage/minor/major/total + road-clear/blocked, misma GSD UAV que tu vuelo | Investigación | Loss unificada + severidad 4 niveles → `collapse_frac` medido reemplaza supuesto 0.3 en `casualties.py` |
| **EBD / xBD Tier 3** | Más sismos fuera Tier 1 | CC BY-NC-SA | Más eventos para split-por-desastre |
| **LEVIR-CD+ / WHU-CD** | Change detection edilicio sin desastre | Abierta | Pre-entrenar siamés en "cambio cero" + hard negatives (misma zona, otra fecha) → falso de 1.23% a ~0 |
| **OpenEarthMap** 5000 imgs 0.25-0.5m 8 clases 97 regiones | Landcover global, generaliza mejor que LoveDA solo | Mixta (mayoría CC BY-NC-SA) | Pre-entreno terreno o joint-training LoveDA+OEM → agua turbia/barro (tu limitación `agua-turbia`) |
| **Sen1Floods11** 4831 chips 512px S1+S2, CC BY 4.0 | Flood + permanent water, 11 eventos | CC BY 4.0 | Solo subset S2-óptico para flood specialist (SAR no te sirve a bordo) |
| **WorldFloods v2** 509 eventos S2, masks EMS/UNOSAT | Flood óptico masivo | Abierta | Aumento flood specialist |
| **FLAME1/2/3 + Boreal Forest Fire 4954 imgs + AusSmoke 15K + MultiNatSmoke 70K + SmokeViz 160K GOES** | Fuego/humo segmentación UAV + satélite | CC BY 4.0 / MIT (verificar cada uno) | **Nuevo head fuego/humo** (ver §6) |
| **WorldPop / GHSL / HRSL** | Densidad hab/km² por pixel | Abierta | Reemplaza `--pop-density 1500` fijo en `casualties.py` por valor predio |

Orden: BRIGHT-óptico + RescueNet primero (mismo dominio edificio), luego OpenEarthMap para terreno, luego fuego/humo.

**5.3. Arquitectura daño para IMX500:**
Siamés actual (2 entradas, 54MB) es inconvertible. Opciones:

- A: single-frame damage (damage3/v3) miniaturizado a MV3-S 192px → 1 entrada, convertible. Pierde pre/post pero gana deploy.
- B: BIT/ChangeFormer-tiny con backbone MobileNet (change detection real, ONNX exportable). Reemplaza siamés reutilizado de DeepLab.
- C: mantener siamés solo en post-vuelo PC (ensemble `post_flight.py`), vuelo usa consenso terreno+flood+single-damage.

Recomendación: A para vuelo + C para post. B solo si A no sube de 0.15.

**5.4. SAM2 para etiquetas limpias:**
`facebook/sam2` refina máscaras baja-confianza a nivel objeto → reconstruir `dataset/xbd_masks` + `pseudo/` con teacher corregido. La otra IA lo propuso en `DATASETS-Y-TECNICAS.md:74`, es barato y sube IoU sin cambiar modelo.

## 6. Track T3 — Nuevos desastres que hoy no detectás

**Fuego/humo (pedido explícito, 0 cobertura hoy):**
Nuevo modelo `fire_smoke_mobilenetv3` 2-3 clases (fondo/humo/fuego) 192-224px <3MB. Train en FLAME + Boreal + AusSmoke (15K reales, no sintéticas) → distill a tiny → MCT → `.rpk` o `cv2.dnn`. Telemetría: agregar `fire_pct, smoke_pct` a `CSV_COLUMNS` (28→30, versionar contrato v2→v3 con fallback `None`). Consenso: fuego dispara `alert=1` independiente de daño estructural. Validar que humo no se confunda con `nodata` borde ni agua turbia.

**Inundación mejorada:**
Flood specialist 0.483 ya es lo mejor. Sumar WorldFloods + Sen1Floods11-S2-óptico al `train_flood_specialist.py` (12ép → 20ép) + misma reducción a 224px. Mantener como primer candidato NPU (ya pasa audit).

**Viento/tornado/huracán:**
No hay sensor viento a bordo. Proxy vía BRIGHT-huracán + RescueNet road-blocked (caminos cortados = viento) + `diag` nuevo `VIENTO SEVERO`. Sin modelo nuevo, solo taxonomía.

**Severidad 4 niveles:**
RescueNet da minor/major/total → exponer `danado_pct + severidad {leve,moderado,colapso}` en `summary.json` + `casualties.py` usa `collapse_frac` medido en vez de 0.3 fijo. Mejora estimación pérdidas sin cambiar física.

## 7. Track T4 — Personas/vehículos en IMX500 (ya a medio hacer)

`cansat/imx500.py:232` + `--det-backend imx500` ya implementado contra picamera2 oficial, pendiente validar en HW. Plan:

1. Usar `.rpk` stock `ssd_mobilenetv2_fpnlite_320x320_pp` para baseline: medir detecciones a GSD 7.7cm/px (persona ~6px a 250m, ~60px a 25m). Documentar que a >100m es sub-pixel (ya advertido en `decisiones.yaml:personas-altitud`).
2. Fine-tune YOLO11n en VisDrone (`visdrone_finetune.py`, `visdrone_yolo11.py` existen) → export Ultralytics ONNX → MCT → `imxconv-pt` → `.rpk` custom (tutorial Arducam 2025-11-05). Comparar vs SSD stock.
3. Vuelo: detección solo <50m o en `high_res` sampler (ahorra NPU). Telemetría `personas/vehiculos` ya en CSV.

## 8. Track T5 — Pipeline vuelo para Zero W (sin esto nada vuela)

- Modo vuelo final: `--camera --frames 1000 --no-damage --det-backend imx500 --img-size 224 --enhance` (terreno+flood tiny o solo terreno). Daño completo solo si tiny <3s/f.
- `mission_pipeline.py:1022` partir en módulos (hoy monolito): `capture.py, infer.py, consensus.py, radio.py`. Agregar `--model-id` (hash MODELS.yaml) a telemetría para trazabilidad.
- `adaptive_sampler.py`: recalibrar `HIGH>=0.30/MED>=0.12` con nuevos % (fuego cambia distribución). Agregar `coverage_boost` ya existe.
- `uart_listener.py` + `protocol.py`: agregar auth mínima (HMAC pre-shared) o al menos sequence + reject replay. Hoy inyección `$LB135` trivial.
- Medir y publicar: `time --folder 3f`, `ms_seg/ms_dmg/ms_total` p50/p95 en JSONL, `audit_imx500.py --all`, `validate_int8_mission.py --frames 40`.

## 9. Track T6 — Estación terrena (lo que hay + qué agregar)

Hoy: `web_server.py:927` (stdlib, SSE, `MissionCache`, `normalize_summary` v2), `web-app/` React19+Vite8+Three (Vuelo/Post/Informe/Present/Jurado, `store/mission.ts:297`, `GpsTrack`, `Corridor`, `Detail` con tabs vis/ens_seg/enhanced/high_res, `validate.ts`, degradación elegante), `tools/make_demo_mission.py:881` + `simulacro.py:150` + `smoke-v4.mjs:264` (~35 checks).

Cambios para otra IA (ordenados):

1. Contrato v3: agregar `fire_pct, smoke_pct, severidad, model_id, quant` a `Frame` (`types.ts`), `CSV_COLUMNS` (`web_server.py`), `summarySchema.ts SCHEMA_VERSION=3` con `normalize_summary` backward-compat (ausente→`None`, UI oculta badge).
2. UI nuevos desastres: badges `INCENDIO/HUMO/VIENTO` + donut 4 severidades + filtro `diag` incluye fuego + `Detail` tab `fire` (overlay rojo/naranja) + `Alerts` slide-in por tipo. Reusa `IMG_TABS` + `ImgKey` (hoy incompleto: omite `full_res|thumb` → fix).
3. Calibración visible: panel Muestreo muestra `uncert p50/p95, ms_total p95, nodata p95, damage-threshold actual` desde `/api/samples` (ya implementado, no documentado en `README-web.md §4` → documentar + `/api/events` SSE).
4. Trazabilidad modelo: KPI `model: terrain_v2 224 INT8 (7a49d4)` + link `MODELS.yaml` hash + `validacion_int8` estado. Evita citar 4 mIoU distintos como antes.
5. Post/Informe: galería `fire_seg`, mapa corredor con segmentos fuego (rojo) + veredicto `SUELO EXPUESTO` ya existe, agregar `HUMO/Agua` veredicto (hoy agua cae en SALUDABLE, discutible ante jurado).
6. Robustez: fix `entrega/b5_seg` invisible (web solo mira `ens_seg` → unificar a `--ensemble` siempre), `corridor_map` dual (`outputs/` vs `entrega/`), `MissionCache._dir_sig limit 512` (falla si >512 ficheros), `static_root` fallback si `dist/index.html` corrupto.
7. Docs: `README-web.md` dice 33/35 aserciones (real ~35-38), `SHOTS` va a `os.tmpdir` no a `tools/shots/`, `make_demo_mission` docstring "22 col." (son 28) + "tira horizontal" (real vertical). `CHECKLIST-SIMULACRO.md` tabla vacía → llenar 4 variantes obligatorias.
8. Demo: extender `make_demo_mission.py` mundo con zona quemada/humo + `simulacro.py --rafaga` inyecta `cap_9000 INCENDIO`. Smoke E2E debe derivar `EXP` del CSV (ya lo hace, mantener).

## 10. Plan de ejecución para otra IA (fases, done criteria)

| Fase | Días | Tareas | Done |
|---|---|---|---|
| F0 higiene | 1 | Re-export opset17+inline, borrar basura §3, fix `export_b5_640`, licencias | `audit --all` verde vuelo, `pytest -q` 162 pass |
| F1 terreno-tiny | 3-4 | Matriz 320→192, MV3-S + PP-LiteSeg-T, KD B5, QDQ+MCT, `imxconv-pt` → `.rpk` terreno | mIoU ≥0.48 a 224px, <8MB, <2s/f Pi o <50ms NPU |
| F2 daño | 5-7 | Split-desastre, RescueNet loss, BRIGHT-óptico + RescueNet train, SAM2 relabel, calibrar umbrales | IoU dañado ≥0.20 cross-event (2x hoy), FP pre==post <0.5% |
| F3 flood+fuego | 3-4 | Flood +WorldFloods/Sen1-S2, nuevo fire_smoke tiny 192px, contrato v3 | IoU flood ≥0.50, fire IoU ≥0.40 val, alert fuego funciona E2E |
| F4 IMX500 | 2-3 | Edge-MDT Linux, convertir terreno+flood+fire, validar `imx500 --seconds 10` en HW | 3x `.rpk` corren on-sensor, `parse_ssd_output` OK |
| F5 vuelo | 2 | Refactor pipeline, modo Zero W final, medir s/frame, UART real Pi↔ESP32 | 10-15 frames en 1-2min, `telemetry.csv` 30 col OK |
| F6 estación | 3 | Contrato v3, UI fuego/severidad, panel calibración, fix `b5_seg/ImgKey/docs`, demo quemado | `npm run smoke` 0 console.error, checklist 12 pasos lleno |
| F7 informe | 2 | `generate_report.py` con números nuevos, `reporte_pruebas.md`, DPD trazabilidad | Solo números de `outputs/metrics/*.json`, licencias tabla OK |

Comandos canónicos que la IA debe usar: `python export_onnx.py --arch ... --opset 17`, `python tools/onnx_inline.py outputs/X.onnx --out-dir dist_pi/outputs`, `python audit_imx500.py --all`, `python eval_sliding.py --split val`, `python validate_int8_mission.py --frames 40`, `python mission_pipeline.py --folder tiles --frames 2 --no-detect --no-damage`, `python tools/make_demo_mission.py --clean && python tools/make_demo_mission.py`, `npm run smoke`.

Riesgos: IMX500 8MB puede forzar 192px + INT8 con caída 3-5 pts (aceptar); BRIGHT-óptico requiere descarga guiada parcial; xBD-NC-SA hereda a pesos daño (declarar, no distribuir comercial); ARMv6 sin NEON hace `cv2.dnn` lento aunque el modelo sea tiny (NPU es el camino real).

Preguntas para vos antes de implementar: 1) ¿Prioridad fuego/humo vs subir daño de 0.10 a 0.20? 2) ¿Aceptás input 224/192px aunque pierda 2-3 pts mIoU? 3) ¿Un solo modelo multitarea (terreno+flood+fire, 1 inferencia) o 3 tinys separados? 4) ¿Tienen PC Linux Python 3.11 para Edge-MDT o lo hago solo `cv2.dnn` por ahora?

---

# APÉNDICE V — Verificación exhaustiva: rotos, a medio hacer, shims y cabos sueltos

## V1. Shims que ya delegan pero siguen ensuciando (BORRAR tras fix hints)

> ✅ **RESUELTO (2026-09-21)**: los 9 shims fueron eliminados
> (`export_v2.py`, `export_b5_512.py`, `export_b5_640.py`, `export_damage_onnx.py`,
> `export_damage_v3.py`, `evaluate_val.py`, `eval_onnx_gap.py`,
> `eval_int8_cpu.py`, `crf_refine.py`) y los hints de `mission_pipeline.py`
> ahora apuntan a `export_onnx.py`. Lo de abajo queda como registro histórico.

- `export_v2.py:24-36`, `export_b5_512.py:26-37`, `export_b5_640.py:30-41`, `export_damage_onnx.py:24-35`, `export_damage_v3.py:26-36` → todos delegan a `export_onnx.py:111-259` (canónico, `OPSET=17:56`, `dynamo=False:201`, `audit:242-250`).
- `evaluate_val.py:1-25` (100 imgs + mIoU 52.44% hardcodeado), `eval_onnx_gap.py:1-26`, `eval_int8_cpu.py:1-25` → delegan a `evaluate.py:259-374`.
- `crf_refine.py:1-33` (`sigmaColor=0.051` erróneo, `I` sin usar) → shim de `cansat/crf.py`.
- `export_b5_640.py:13-20` original roto (cargaba `best_terrain_b5.pth` inexistente + clase `SegformerTerrain` inexistente). Causó que `post_flight.py` salteara 2ª pasada en silencio.

**Hints rotos a arreglar antes de borrar:**

- `mission_pipeline.py:614` dice `Corré export_v2.py` → `export_onnx.py`
- `mission_pipeline.py:635-637` dice `export_damage_onnx.py / export_damage_v3.py` → `export_onnx.py --arch damage`
- `post_flight.py:199` dice `python export_b5_512.py` → `export_onnx.py --arch segformer`
- `quantize_onnx.py:45-46` default `cansat_seg_deeplabv3plus_mobilenetv2.onnx` → `cansat_seg_terrain_v2.onnx` (vuelo)
- `docs/reporte_pruebas.md:88` nota "solo export_onnx y export_b5_640 pasaban dynamo=False" es historia pre-shim → actualizar.

## V2. `legacy/` verificado (10+README, NO tocar lógica, solo no resucitar)

`legacy/README.md:17-28` + `MODELS.yaml:274-296`:

- `test_satlas.py / compare_satlas.py`: typo `SwimB_MultiTask`, asume dict, lee imagen inexistente, descarga `torch.hub` en runtime.
- `train_qat.py`: QAT eager deprecado torch 2.x + `qconfig fbgemm` x86 inútil en ARM + mIoU 0.5317 constante.
- `mae_pretrain.py`: entrenaba sobre `loveda_raw` completo incluido Val + guardaba encoder en raíz + stride 32 vs OS8.
- `train_v3.py:65-172`: pseudo `dataset/pseudo/img320`, BN congelado, exportaba `_v4.onnx` por error.
- `train_v4.py:33-197`: `AugDataset:71-80` etiquetas desalineadas 1/3, `BoundaryLoss:110-121` no-diferenciable (arreglados en código, checkpoint arrastra defecto), `torch.save` crudo `:180` sin `save_ckpt` + exporta última época no BEST, `torch.load` sin `weights_only` (`v3:81,84`, `v4:136`) vs `export_onnx.py:124`.
- `mae_mobilenetv2_encoder.pth`, `yolo26n.pt`: sin referencia, vuelo es `models/yolov8s.pt`.

## V3. `pi/` + quantize (roto/desactualizado real)

- `pi/convertir_modelos.sh:5` convierte la base vieja `cansat_seg_deeplabv3plus_mobilenetv2.onnx` (45.76%, EXPERIMENTAL) no el vuelo `cansat_seg_terrain_v2.onnx`. Sintaxis `imx500_converter --help` no verificada → fijar o marcar NO SOPORTADO.
- `pi/instalar_en_pi.sh:1-90` sano. `pi/preparar_deploy.ps1:36-43` por defecto solo terreno+flood, `-Todos` agrega resto pero `dist_pi/` en disco tiene 2/5 inline → regenerar.
- `quantize_all.py:33-39` dinámica (solo pesos, docstring ya admite ganancia nula) vs `quantize_onnx.py:157-164` QDQ estático nunca ejecutado. `MODELS.yaml:251-272`: dinámico 23-26x más lento ORT, no carga en cv2, acuerdo 90.5%<95%. Vuela FP32 hasta re-medir. `mission_pipeline.py:451-456` ya no tiene `--int8` → aclararlo.

## V4. Trains (detalles que la IA debe saber)

- `train.py:261-263` 512px/32bs/40ép vs `train_terrain_v2.py:123-128` 320px/8bs/12ép (vuelo) vs `train_cbam.py:90-93` 320px/8bs/10ép + init distilled.
- Pesos: `train.py:201-226` con caché huella vs `v2:100-110` `1/sqrt` clip 0.3-8 sin caché. `train_cbam.py:110-111` `num_workers=4` hardcodeado rompe en Windows → 0 o flag. `train_cbam.py:128` `GradScaler("cuda")` → con `enabled=` como `train.py:298`.
- `train_cbam.py:106-107` ya usa `load_into(min_frac=0.5)`, bien.

## V5. Tests + CI (agujeros)

> ✅ **PARCIALMENTE RESUELTO (2026-09-21)**: el CI estaba muerto (workflow en
> `cansat_seg_poc/.github/`, GitHub solo corre `.github/workflows/` de la raíz)
> y ahora corre 3 jobs verdes. Cobertura agregada: `post_flight.resolve_b5`,
> `sliding_logits` con frame < ventana, `build_evidence`, `detect_objects` con
> stub, `terrain_percentages`, `_green_patches`, `enhance_image.ensure_model`
> (sin red) y el guard de `cv2.dnn_superres`, `quantize_onnx` QDQ estático
> end-to-end con un modelo de juguete, y `export_onnx` en tests `slow`
> (torch). Sigue sin cubrir: entrenamiento real, YOLO con ultralytics,
> `picamera2`/UART (requieren hardware o datasets).

- 12 tests solo `cansat.*` + geometría + sampler. **Cero cobertura:** entrenamiento, `export_onnx.py:111-259`, quantize, YOLO/VisDrone, `picamera2`/UART real, B5 fallback `post_flight.py:80-85`, EDSR `enhance_image.py:55-157` (requiere `contrib` + red). Marcas `slow/gpu/dataset` declaradas en `pyproject.toml:103-106` pero ningún test las usa.
- `.github/workflows/ci.yml:72-93` job `frontend` **siempre rojo**: `working-directory: estacion_terrena/web-app:77` no existe (`Test-Path False`). Eliminar o re-agregar.
- Duplicado a hashear: `models/yolov8n.pt` vs `yolov8n_base.pt` mismo tamaño 6549796 + `yolo11n.pt` raíz vs `models/yolo11n_visdrone.pt`.

## V6. Estación terrena (fixes con línea)

- `README-web.md:89-97` tabla API omite `GET /api/samples` (`web_server.py:804-810`) y `GET /api/events` SSE (`:812-814`, `:676-728`). Docstring `web_server.py:7-16` igual. Agregar 2 filas.
- `types.ts:8 vs :12-19`: `ImgKey` 4 vs `FrameFiles` 6 (`+full_res,thumb`). `vocab.ts:56-61 IMG_TABS` 4 → `Detail.tsx:46` nunca ofrece esas tabs (solo fallback en `Corridor.tsx:95`, `poster.ts:104`, `Presentation.tsx:94-95`). Decidir: ampliar o renombrar a `DetailImgKey`. Ojo: ampliar rompe `smoke-v4.mjs:125,130` (espera 4/2 tabs).
- `web_server.py:309` default `schema_version 1` vs `summarySchema.ts:14` + `make_demo_mission.py:810` + `entrega/summary.json:2` = 2 → cambiar a 2.
- `make_demo_mission.py:14` "22 columnas" falso, real 28 (`:759-763` 23+5 DPD = `web_server.py:63-71`). `README-web.md:71,80-82` dice 23. Unificar a `28 (23+5 DPD)`. Duplicada `"_cls_res":726,728` → borrar una.
- Corredor: `make_demo:19,584-608` horizontal autoconsistente pero `Corridor.tsx` + `tour.ts:11` vertical 250→0m, `ReportView.tsx:206-211` lo imprime minúsculo. Decidir orientación única.
- `smoke-v4.mjs:15` `SHOTS=os.tmpdir()` vs README "`tools/shots/`". Conteo real `grep check\(` = **41**, no 33/35 (`README:128,180,219`). `tools/shots/r4/` inexistente aunque `README:275` + `shots-r4.mjs:4` lo referencian.
- `CHECKLIST-SIMULACRO.md:25-38` 12 checks OK, `:50-55` 4 variantes obligatorias OK, `:59-61` tabla vacía (0 corridas) → TODO operador.
- `web-app/package.json:28` `shadcn ^4.21.0` nunca importado (solo comentario `Gloss.tsx:1`) → eliminar o documentar. `lucide-react ^1.45.0` **sí es correcto** (instalado 1.45.0, major 1.x existe) → corregir supuesto del plan, no código.
- Sin repo git (`fatal: not a git repository`), `.gitignore` no vigente hasta `init`. Mantener ambos `package-lock.json`, excluir `dist/node_modules/shots/outputs/entrega`.

## V7. Medir antes de afirmar (lista de PENDIENTEs)

- `segformer_b5 miou PENDIENTE` (`MODELS.yaml:224`, `.data` 338MB), YOLO P/R NO MEDIDO (`:198`), `--det-backend imx500` no validado HW (`guia_pi.md:147-148`), s/frame Zero v1 (`:120-130`), EDSR min/frame + `contrib` (`post_flight.py:227-228,285-318`), daño split-fila→desastre (`train_damage.py:77,v2:89,v3:84,siamese:226` → `cansat/xbd.py split_por_desastre` + `tests/test_xbd.py`).

Con esto la IA tiene **cero dudas**: qué borrar, qué unificar, qué arreglar con línea exacta, qué medir en placa, y qué decidir (input 192/224, 1 vs 3 modelos, orientación corredor, ImgKey, fuego vs daño prioridad, Linux Edge-MDT disponible o solo cv2.dnn).

---

# V8 — Verificación final contra el DPD (2026-09-18, con PDF leído)

Fuente: `La Base - DPD CANSAT 2026.pdf` (Equipo 135, E.E.S.T. N.º 4 El Palomar). Lo que sigue es lo que el DPD **sí exige** y el proyecto aún no cierra del todo, más un matiz que relaja el plan.

## V8.1. Matiz importante: el DPD NO exige IA en tiempo real a bordo

- DPD, Diseño preliminar / Cómo funciona: la Pi Zero 2 W "captura... asocia con telemetría... y procesa en tiempo real", pero la Secuencia de Operaciones aclara que durante el descenso la Pi "captura... asocia... y almacena para su posterior procesamiento", y que **"una vez recuperado el CanSat, las imágenes serán mejoradas mediante IA y segmentadas"** para generar mapas.
- Consecuencia: el flujo **post-vuelo en PC (B5 + EDSR + ensemble + `post_flight.py` + estación Post/Informe) ya cumple la misión secundaria ante el jurado**. Los tracks T1 (tiny) y T4 (IMX500) son optimización de vuelo, no condición de cumplimiento. La IA debe priorizar F7 (post + informe con números de `outputs/metrics/*.json`) al mismo nivel que F1/F4. No sobre-ingenierizar el onboard a costa del informe.

## V8.2. Humedad BME280: se mide pero nunca viaja (gap real)

- DPD Subsistemas: BME280 mide presión, temperatura **y humedad**; la Pi asocia "hora, posición, altitud, presión, temperatura, etc." a cada imagen.
- Verificado por búsqueda: cero manejo de humedad en `cansat/` (`protocol.py`, `summary.py`, `casualties.py` sin campo `hum`), ni en `CSV_COLUMNS` (`mission_pipeline.py:133`), ni en estación.
- Acción para la IA: agregar `hum_pct` al contrato v3 junto a `fire_pct/smoke_pct` (28→31 col., `mission_pipeline.CSV_COLUMNS` + `web_server.CSV_COLUMNS` + `types.ts Frame` + `summarySchema` + `normalize_summary` backward-compat), o documentar en `docs/ALINEACION-DPD.md` por qué se descarta (la primaria solo exige presión+temperatura; la humedad es bonus). Costo bajo, cierra trazabilidad sensor→frame→estación.

## V8.3. IMU MPU6050: eyección/apogeo/aterrizaje fuera del dominio Pi (aclarar, no codificar de más)

- DPD: eyección por variación de altitud/presión en apogeo (~250 m), descenso 2-4 m/s, aterrizaje detectado por IMU (fin de aceleración/vibración), test de vibraciones con plataforma improvisada.
- Hoy: apogeo es parámetro `--apogee`, aterrizaje es heurística de estación (`missionState.ts:23-24`, `last<=3` sin summary), sin campos IMU en telemetría (búsqueda `imu|MPU|accel` sin matches funcionales).
- Acción: **no** meter la IMU en la Pi. La IMU queda en dominio ESP32; la Pi solo necesita (a) bandera `landed` opcional en protocolo v3 si la ESP32 la provee, o (b) mantener heurística alt + documentarla. Sí incluir en plan no-software: fijación microSD (silicona/kapton), alivio de tensión JST, test de vibración con telemetría estable (DPD lo pide explícito).

## V8.4. Sustitución de hardware: hay que justificarla por escrito (Zero 2 W → Zero W v1, Module 3 → IMX500)

- DPD Presupuesto confirma compra: Zero 2 W $154.754 + Camera Module 3 $132.500; masa total 223 g con Zero 2 W de 11 g.
- Decisión actual (usuario): volar **Pi Zero W v1 ARMv6 + IMX500**. `docs/ALINEACION-DPD.md §3` ya lo registra, pero ante jurado hace falta addendum:
  1. Motivo (disponibilidad/costo/potencia: Zero W consume menos, ayuda a batería 18650 + step-up 1 A y a masa 223 g).
  2. Equivalencia funcional: IMX500 on-sensor reemplaza YOLO en Pi (`cansat/imx500.py`, `--det-backend imx500`); `cv2.dnn` reemplaza ORT; salidas (telemetría 28/31 col., `summary.json` v2/v3, corredor, evidencias) idénticas.
  3. Evidencia medida: s/frame en placa + detecciones IMX500 a GSD 7.7 cm/px + validación `audit --all`.
- Acción: crear `docs/SUSTITUCION-HW.md` (1 página) y referenciarlo en `ALINEACION-DPD.md §3` y en el informe. Sin esto, el jurado puede marcar desvío del DPD.

## V8.5. Tests de componente del DPD que el software no cubre (procedimiento, no código)

- BME280 vs estación meteorológica calibrada; GPS TTFF + precisión posición/altitud (ATGM336H); LoRa a distancia real **con CanSat cerrado** (la carcasa afecta alcance); cámara con distinta luz/movimiento; almacenamiento prolongado (integridad + sincronía imagen↔metadatos); consumo total con batería definitiva (el pipeline mide `ms_total`, no mA → medir en banco con USB-tester).
- La IA no debe codificarlos: debe dejar procedimientos + bitácora. `EstacionTerrena/CHECKLIST-SIMULACRO.md` (12 pruebas integridad 1-8 + 4 variantes) es el lugar; su tabla de corridas está vacía (`:59-61`) → llenar en campaña. Roles DPD (Juan Manuel y Alan = estación terrena) ya coinciden con el repo.

## V8.6. Post-recuperación: falta la herramienta SD→PC (gap pequeño)

- DPD Operación: tras recuperar, descargar microSD, verificar integridad, respaldar, y recién ahí procesar.
- Hoy `post_flight.py` asume `outputs/mission/` ya en PC. Acción: script/checklist `ingest_sd.sh` (montar SD, `rsync vuelos/<TS>/`, `sha256sum`, `verify_dataset`-like de pares imagen↔telemetría) + documentarlo en `pi/guia_pi.md`. 2-3 horas de trabajo, evita perder la misión por SD corrupta (brownout: ya previsto 470 µF + 100 nF en `docs/ELECTRONICA-PCB.md`).

## V8.7. Qué es extensión (no exigible por el DPD)

- Venezuela = inundación/deslizamiento → justifica priorizar flood specialist. Fuego/humo, viento/tornado, severidad 4 niveles, WorldPop por predio: valiosas como extensión, **no** las pide el DPD (que pide vegetación/personas/edificios/agua/otros + estrés + daños + pérdidas). Que la IA las marque como "extensión" en el informe para no rendir cuentas de más.
- Difusión (redes, video `youtu.be/J84qZzx0qzw`, cuaderno de campo, foto/video por Luz): entregable no-software; la IA solo debe no romperlo (export CSV/poster PNG para difusión ya existen en la estación).
