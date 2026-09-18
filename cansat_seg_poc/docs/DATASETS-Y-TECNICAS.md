# Datasets y técnicas para mejorar el sistema — CanSat LB135

Investigación del 2026-09-17. Responde a: *"buscar nuevas técnicas de
entrenamiento (aunque requieran entrenar desde cero)"* y *"existen otros
datasets para mejorar los desastres (tornados/huracanes, terremotos)"*.

---

## 1. Estado actual medido (punto de partida)

| Modelo | Métrica honesta | Nota |
|---|---|---|
| Terreno (DeepLabV3+ MV2, 5 clases) | **mIoU 52.19 %** (Val completo) | fuerte para edge; margen de mejora acotado |
| Daño principal (3 clases, xBD) | IoU "dañado" **0.104** en desastres no vistos (0.132 en train) | el número viejo (0.32) tenía fuga |
| Daño two-stage | **0.025** con desastres no vistos; 0.436 con edificios GT | vive de la máscara de edificios |
| Siamés (re-entrenado) | DAÑADO* **0.601**; 1.23 % de daño falso con pre==post | desactivado por defecto |
| Flood specialist (re-entrenado) | IoU flood **0.483** | remapeo corregido |
| Personas | sin métrica todavía | el SSD del IMX500 y YOLO existen; falta medir |

**Conclusión**: el eslabón débil no es el terreno (52 % es competitivo), son los
modelos de daño y la estimación de personas. Ahí van las mejoras.

## 2. Datasets recomendados (además de los actuales)

| Dataset | Qué aporta | Licencia | Encaje |
|---|---|---|---|
| **BRIGHT** (Chen et al., ESSD 2025, Zenodo `14619797`, GitHub `ChenHongruixuan/BRIGHT`) | Primer dataset multimodal VHR **óptico + SAR** de daño edilicio: 14 eventos (sismos, huracanes/ciclones, incendios, inundaciones, volcanes, explosiones, conflictos), 23 regiones, >380 000 edificios, 0.3–1 m/px. Benchmark oficial del IEEE GRSS DFC 2025 Track II, con **baselines de transferencia cross-evento**, domain adaptation no supervisada y change detection multimodal | abierta (algunos sectores ópticos requieren descarga guiada) | ⚠ **VERIFICADO 2026-09-18 (descarga real)**: el benchmark `pre-event.zip`/`post-event.zip`/`target.zip` es **pre-óptico + post-SAR** — el loader oficial apila el gris a 3 canales para el post (`make_data_loader.py`) y las etiquetas son 0=fondo, 1=intacto, 2=dañado, 3=destruido (`cvprw26/src/dataset/classes.py`). **NO sirve para el modelo de vuelo, que es 100 % óptico** (post descargado: 4246 `.tif` 1024² en escala de grises). Sólo aplicaría a un experimento cross-modal (SAR→óptico), fuera del alcance del CanSat. **Reemplazado por RescueNet** como dataset de daño óptico nuevo |
| **RescueNet** (Rahnemoonfar et al.) | Daño edilicio desde UAV post-huracán, 10 903 edificios; propone **arquitectura unificada end-to-end** (segmentación + daño) y una **loss específica**: BCE de localización + CE selectiva sólo en foreground | investigación | **La receta de loss para `train_damage_v3`**: nuestra two-stage ya usa ignore en fondo; RescueNet formaliza y agrega la cabeza de localización |
| **EBD / xBD Tier 3** | Sismos y eventos extra fuera del Tier 1 de xBD | xBD: CC BY-NC-SA 4.0 | Más eventos de sismo para el protocolo cross-event |
| **VisDrone** (ya usado) | Personas/vehículos desde dron | challenge | Fine-tune del detector; ya hay scripts (`visdrone_finetune.py`) |
| **HERIDAL / SARD** | Búsqueda y rescate: personas en escenas aéreas | investigación | Sólo si se quiere apuntar a "personas perdidas"; GSD de CanSat (≈8 cm/px) es mejor que la de esos sets |
| **WorldPop / GHSL / HRSL** | Grids de población (hab/km²) | abierta | Insumo directo de `cansat/casualties.py`: reemplaza la densidad fija por el valor del predio |
| **Sen1Floods11 / WorldFloods** | Inundaciones con Sentinel-1/2 | abierta | Alternativa a FloodNet con etiquetas íntegras (FloodNet tiene clases solapadas) |
| **LEVIR-CD+ / WHU-CD** | Change detection edilicio | abierta | Pre-entrenamiento del siamés con pares reales sin desastre (aprende "cambio cero") |

## 3. Técnicas de entrenamiento recomendadas (ordenadas por retorno)

### 3.1 Alto retorno, bajo costo (implementables ya)

1. **Split por desastre en los 4 scripts de daño** (`train_damage*.py`,
   `train_siamese_damage.py`). Es la causa principal del 0.32 inflado.
   `eval_sliding.py` ya tiene el helper (`split_por_desastre`).
2. **Loss estilo RescueNet** para `train_damage_v3`: BCE de localización +
   CE selectiva en foreground. Nuestra versión actual (ignore_index=0) es una
   aproximación; la de RescueNet mejora la clasificación de clases intermedias
   en un orden de magnitud según el paper.
3. **Pares negativos** en el siamés (ya implementado: `--neg-frac 0.3`). Bajó el
   daño falso de 5.5 % a 1.23 %; subir a 0.5 y agregar *hard negatives*
   (mismo sitio, otra fecha sin desastre, de LEVIR-CD+) puede llevarlo a ~0.
4. **Pseudo-etiquetado con los modelos corregidos**: regenerar
   `dataset/pseudo` con el flood re-entrenado y el terreno v2 real
   (`pseudo_label.py`); hoy las máscaras de agua salieron del flood roto.
5. **Augmentación de dominio de vuelo**: ya existe `degrade_flight`
   (down-up + blur + ruido) y jitter. Agregar **rotación arbitraria** (el
   CanSat gira bajo paraguas) y **perspectiva leve** (frames oblicuos).
6. **Calibración de umbrales** con el JSONL de vuelo: `--damage-threshold` ya es
   configurable y cada frame registra `danado_*`, `uncert`, `nodata_pct`.

### 3.2 Alto retorno, costo medio

7. **Destilación desde SegFormer-B5** (ya implementada, `distill_terrain.py`):
   con los fixes de esta sesión (pesos de clase del dataset actual, KD
   enmascarado con ignore, semilla, export del mejor) puede recuperar 1–3 pts
   sobre el student sin cambiar el costo de inferencia. Medir con `evaluate.py`.
8. **Self-supervised pretraining** del backbone: hay `mae_pretrain.py` (MAE
   propio) y la alternativa moderna es **DINOv2/v3** (features congeladas +
   cabeza liviana) o **SatMAE++ / Scale-MAE** para dominio satelital. Para
   *nuestro* caso el techo es el mIoU de LoveDA (~52 %), así que conviene
   medirlo antes de invertir.
9. **Change detection moderno para el siamés**: `BIT` (transformer bi-temporal)
   o `ChangeFormer` con backbone MobileNet, exportables a ONNX. Reemplazan al
   siamés actual (features de un DeepLabV3+ reutilizado) por un detector de
   cambio real entrenado con pares.
10. **SAM2 para pseudo-etiquetar** (`facebook/sam2`): dada una máscara de baja
    confianza del modelo, SAM2 la refina a nivel objeto. Muy útil para
    construir el dataset de daño con etiquetas limpias a partir de xBD +
    predicciones.

### 3.3 Rediseño (sólo si sobra tiempo)

11. **Arquitecturas edge**: PIDNet-S (78.6 mIoU Cityscapes, 7.6 M params) y
    PP-LiteSeg/STDC superan a DeepLabV3+ en la relación precisión/latencia en
    GPU, pero **en ARMv6 con `cv2.dnn` la compatibilidad de ops no está
    garantizada** (hay que exportar y probar `cv2.dnn.readNetFromONNX` antes).
    MobileNetV4-Conv-S promete latencia ~2× mejor que V3 en CPUs móviles, pero
    estudios controlados 2025 muestran que la ventaja no es universal.
    **Decisión sugerida: no cambiar de arquitectura antes de la competencia.**
12. **Entrenamiento semi-supervisado (UniMatch V2)**: con DINOv2 como encoder
    supera al supervised con 1/8 de las etiquetas. Es la vía si se quiere subir
    mIoU de terreno sin etiquetar más, pero implica un stack nuevo.
13. **Conversión al NPU del IMX500**: convertir el modelo de terreno con Sony
    Edge-MDT (MCT → `imxconv-pt` → `imx500-package`). Requiere PC con ≥4 GB y
    Python 3.11; la segmentación semántica no está en la lista soportada de
    Ultralytics, pero el converter acepta modelos ONNX custom (opset 15-20) si
    las ops están soportadas. **Es el camino para que la Zero v1 vuele rápido.**
14. **QAT / INT8 estático (QDQ)**: `quantize_onnx.py` está escrito y nunca se
    ejecutó. Sólo tiene sentido si se valida en la placa; el INT8 dinámico
    actual es inútil (ver `MODELS.yaml → validacion_int8`).

## 4. Estimación de pérdidas humanas: qué hace el sistema hoy

`cansat/casualties.py` implementa un **modelo de exposición** explícito:

```
personas_expuestas  = densidad_poblacional × área_relevada × ocupación
personas_afectadas  = personas_expuestas × fracción_de_área_dañada
pérdidas_estimadas  = personas_afectadas × fracción_de_colapso × letalidad_en_colapso
```

Supuestos por defecto (todos configurables): 1500 hab/km², ocupación 0.6,
colapso 0.3, letalidad 0.1; banda de incertidumbre ×0.5–×2.

**Cómo se compara con el estado del arte**: los sistemas operativos reales
(PAGER del USGS, GRADE del Banco Mundial, UNOSAT) usan modelos de fragilidad
por tipo estructural, intensidad del evento y hora; la práctica 2025 publicada
para el sismo de Myanmar combina InSAR + huellas edilicias + WorldPop +
funciones de mortalidad empíricas. Nuestro modelo es una versión de primer
orden de esa cadena, adaptada a lo que un CanSat puede medir: **área dañada
por frame + densidad del predio**. Para acercarlo al estado del arte:

1. Cargar la densidad real del predio (WorldPop API o dato INDEC) en
   `--pop-density`.
2. Si se consigue distinguir severidad (colapso vs daño leve), usar el
   `collapse_frac` medido en vez del supuesto.
3. Documentar el rango en el informe (ya se publica con banda).

## 5. Plan sugerido (2 semanas) — estado al 2026-09-17

| Día | Tarea | Estado |
|---|---|---|
| 1–2 | Split por desastre en los 4 scripts + re-entrenar daño con la loss de RescueNet; medir con `eval_sliding --split val` | ✅ **hecho**: helper `cansat/xbd.py`, split aplicado en `train_damage{,v2,v3}` y `train_siamese_damage`, loss `--loss rescue` implementada y en entrenamiento |
| 3 | Regenerar pseudo-labels con flood/terreno corregidos; re-correr destilado | ⏸️ pendiente (la línea v3/v4 quedó en `legacy/`, así que es opcional) |
| 4 | Validar IMX500 en la Pi + medir s/frame y detecciones de personas | ⏸️ bloqueado: falta la microSD/AI Camera en mano |
| 5 | Calibrar `--damage-threshold` y `--uncert-max` con el JSONL del simulacro | ✅ herramienta creada: `tools/calibrate_thresholds.py` (usa xBD val con split honesto o frames propios con `--folder/--labels`); falta correrla con frames del predio |
| 6–7 | (Opcional) BRIGHT subconjunto óptico para protocolo cross-event | ❌ **descartado 2026-09-18**: verificado que BRIGHT es pre-óptico + post-SAR; no aplica al modelo óptico de vuelo. RescueNet (UAV óptico) pasa a ser el dataset nuevo de daño |
| 6–7 (bis) | **F2 con RescueNet** (UAV óptico): tiles + remapeo + entrenamiento mezclado | ✅ **hecho 2026-09-18**: `prepare_rescuenet.py` (17 560 tiles train + 2 093 val a 640², remapeo a other/intacto/dañado), `train_damage_v3.py --extra-manifest/--xbd-repeat`, 3 modelos medidos (ver «F2 — RescueNet») |
| 8–10 | EDSR en vuelo, informe con números nuevos, ensayo completo | 🟡 el informe se regenera solo (`generate_report.py`); EDSR ya funciona offline |

### Mejoras ya aplicadas fuera del plan

- **Detector de personas**: default cambiado a `yolov8n_visdrone.pt`. Medición
  sobre imagen aérea: **254 personas vs 42** del COCO yolov8s (los COCO pierden
  personas sub-píxel). `yolo11n_visdrone.pt` equivalente (251).
- **Loss de RescueNet**: `rescue_loss()` en `train_damage_v3.py` (BCE de
  localización edificio/fondo + CE selectiva en foreground, adaptada a una
  sola cabeza de 3 clases).
- **Métrica de selección**: los scripts de daño ahora eligen el mejor checkpoint
  por IoU de daño **sobre edificios** (la métrica de la misión), no por mIoU de
  3 clases dominado por el fondo.

### Resultados medidos (2026-09-17)

| Modelo | Antes | Ahora | Protocolo |
|---|---|---|---|
| Daño principal (v2) | 0.053* / 0.381* (con fuga) | **0.107 / 0.215** | crops nativos / tile completo, desastres NUNCA vistos (A/B sobre 40 tiles) |
| Daño two-stage (v3) | 0.025 | **0.18-0.26 en la 1.ª época** (12 épocas en curso) | ídem, con la loss de RescueNet |
| Umbral de daño | 10 % (sin calibrar) | **11.8 %** (F1 0.77, FPR 4.3 %) | `tools/calibrate_thresholds.py` sobre 120 tiles de val honesto |
| Detector de personas | 42 detecciones (COCO) | **254** (VisDrone) | `dataset/pruebas/PERSONAS_AUTOS.png` @1280 |

*El 0.381 del modelo viejo en tile completo es memorización: entrenó con ~90 %
de cada desastre (split por fila). No es un número citable.

### F2 — RescueNet (2026-09-18): trade-off de dominio

Mismo protocolo two-stage (IoU de dañado sobre edificios) en los dos val:

| Modelo | xBD held-out (satélite 0.3 m/px) | RescueNet val (UAV 0.05-0.08 m/px) |
|---|---|---|
| xBD-only (`best_damage_v3.pth`) | **0.472** | 0.497 |
| Mezcla 82 % RescueNet (`best_damage_v3_uav.pth`) | 0.004 | 0.686 |
| Balanceado xBD×4 + 4k RN (`best_damage_v3_bal.pth`, ONNX auditado) | 0.096 | **0.735** |

Los dominios compiten: agregar RescueNet sube el daño UAV de 0.497 → 0.735
pero hunde xBD de 0.472 → 0.096 (olvido catastrófico con 82 % de RescueNet;
parcial con el balanceo ×4). **El vuelo del CanSat es UAV (7.7 cm/px)**, así
que el balanceado es el modelo de vuelo elegido (2026-09-18) y el xBD-only
queda como referencia cross-event del informe. Caveat: el 0.735 de RescueNet
val puede estar inflado por proximidad geográfica train/val del propio dataset
(mismos vuelos post-Michael); el número cross-dataset honesto es el de xBD.

**Calibración en el dominio de vuelo** (`tools/calibrate_thresholds.py
--rescuenet --tiles 600 --torch --min-dano-px 10`, GPU): el two-stage
balanceado tiene F1-óptimo **10.18 %** (F1 0.828, precisión 0.776, **recall
0.887**, FPR 41.5 % sobre val de zona de desastre); el umbral de vuelo quedó en
**10.2 %** (prioriza detección). El principal xBD **satura en UAV** (mediana
62.7 % de "daño" en tiles sin daño): en vuelo no discrimina, el two-stage es el
que decide. Recalibrar con frames reales del predio (`--folder/--labels`)
antes de la campaña.

**Experimento descartado**: re-entrenar el principal (damage3) con la misma
mezcla empeoró ambos dominios (xBD 0.173 → 0.047; RescueNet 0.476), así que el
principal xBD se mantiene como segundo votante del consenso.

**Interpretación honesta**: la generalización cross-event sobre satélite sigue
siendo modesta (IoU 0.10-0.47 según el modelo). Con RescueNet (UAV, el dominio
real del vuelo) el daño sube a 0.735, pero ese mismo modelo deja de servir para
satélite: con esta receta no hay un solo modelo bueno en ambos dominios. Para
el CanSat alcanza para *detectar* daño a nivel frame (F1 0.83, recall 0.89) y
estimar exposición; el mapa fino por edificio sigue pendiente (más datos
etiquetados del dominio UAV, o un modelo por dominio según la GSD del frame).
