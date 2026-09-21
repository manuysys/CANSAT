# CanSat La Base (CONAE 135) — Reporte de pruebas de misión secundaria

Generado: 2026-09-20 por `generate_report.py`.

> Las secciones 1–5 salen de **artefactos medidos** en disco. Las 7–9
> salen de `docs/decisiones.yaml`, que es editable sin tocar código.
> Si falta un artefacto, el informe **dice que falta** en vez de citar
> un número hardcodeado (que era lo que pasaba antes).

## 1. Modelo de segmentación
- DeepLabV3+ MobileNetV2 (output stride 8, ASPP, decoder stride-4), 5 clases.
- Clases: vegetation, building, water, bare_ground, other.
- Entrada: `[1, 3, 320, 320]`.
- ONNX: `outputs\cansat_seg_terrain_v2.onnx` (51.0 MB).
- Precisión de vuelo: **FP32**.

> FP32 confirmado por medición: los `*_int8.onnx` existentes son cuantización
DINÁMICA, que ni siquiera carga en OpenCV DNN (opcode DynamicQuantizeLinear)
y con ORT es 23-26× MÁS LENTA que FP32 en CPU. Detalle en MODELS.yaml →
validacion_int8. El único camino INT8 viable es QDQ estático verificado en
la placa.

## 2. Métricas cuantitativas (Val, modelo ONNX)
- Fuente: `C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\outputs\metrics\val_20260917_200724.json` — generada 2026-09-17T23:07:24.445387+00:00
- Modelo: `outputs/cansat_seg_terrain_v2.onnx`
- Muestra: 1669 imágenes de Val (semilla 42).
- Muestra completa.

| Clase | IoU | Precisión | Recall | Soporte (px) |
|---|---|---|---|---|
| vegetation | 54.8% | 77.0% | 65.5% | 59,836,301 |
| building | 50.5% | 55.9% | 83.8% | 11,995,267 |
| water | 63.9% | 72.4% | 84.5% | 19,487,560 |
| bare_ground | 42.9% | 71.7% | 51.7% | 15,037,742 |
| other | 48.9% | 63.6% | 67.8% | 60,083,495 |

**mIoU: 52.19%** | Pixel accuracy: 68.62% | 166,440,365 píxeles

## 3. Validación de cuantización INT8
- ⚠ **Aún no generada.** Correr:
  ```bash
  python validate_int8_mission.py
  ```
- El 87.6% que figuraba en el informe anterior **no es válido**:
  se midió sobre las evidencias anotadas (overlay + cajas +
  texto quemado), con fórmulas de USI/NDVI distintas a las del
  vuelo, y comparando modelos distintos. Ver el docstring de
  `validate_int8_mission.py`.

## 4. Simulacro de descenso (prueba de integridad N.º 6 del DPD)
- Fuente: `C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\outputs\mission\telemetry.csv` (3 paquetes).
- Altitud: 250 m → 244 m.
- Veredictos: ZONA SALUDABLE: 2, ALTO ESTRÉS URBANO: 1.
- Diagnósticos: SIN DESASTRE: 3.
- Alertas: 0 de 3 frames.
- Sampler: FULL: 3.
- Nitidez (Laplaciano): min 142 · mediana 258 · max 334.

## 5. Evidencias generadas
- Tiles Test segmentados: 10
- Frames de evidencia de misión: 38
- Detecciones YOLO anotadas: 5
- Frames alta prioridad (sampler): 15
- Frames prioridad media: 11
- Thumbnails: 2
- Overlays ensemble (post-vuelo): 12
- Frames mejorados con EDSR: 3
- Simulaciones de desastre: 5

## 6. Datasets y licencias

| Dataset | Uso en el proyecto | Licencia | Restricción |
|---|---|---|---|
| LoveDA (Zenodo 5706578) | Segmentación de terreno (5 clases) | CC BY 4.0 | Atribución |
| xBD / xView2 | Modelos de daño estructural | **CC BY-NC-SA 4.0** | **No comercial + ShareAlike** |
| FloodNet (Kaggle) | Especialista de inundación | sin declarar en la ficha | Verificar antes de distribuir |
| VisDrone | Fine-tune del detector | ver términos del challenge | Verificar |
| Esri World Imagery | `baseline.png` del siamés | términos de ArcGIS Online | Verificar para un entregable |

⚠ **ShareAlike de xBD:** los pesos entrenados con xBD
  (`best_damage*.pth`, `best_siamese_damage.pth` y sus ONNX) son
  obra derivada y heredan la licencia no comercial + compartir-igual.
  Atribuir explícitamente en el DPD y consultar a CONAE antes de
  distribuirlos. Los modelos entrenados **sólo con LoveDA** no tienen
  esa restricción: conviene tener claro cuál es cuál (ver `MODELS.yaml`).

## 7. Decisiones de diseño
- Exportación ONNX con dynamo=False y sin dynamic_axes (requisito del conversor Sony IMX500). audit_imx500.py ahora valida de verdad contra estos requisitos y devuelve exit code != 0 si algo no cumple; antes sólo imprimía el opset y el recuento de operadores. Nota: sólo export_onnx.py y export_b5_640.py pasaban dynamo=False; los otros cuatro exportadores no. Unificado en cansat/onnxio.py + export_onnx.py. Además, audit_imx500.py ya no marca "pesos externos" por la mera presencia de un .onnx.data suelto: mira el grafo (data_location=EXTERNAL).
- La estimación de pérdidas humanas del DPD es un modelo de EXPOSICIÓN con supuestos declarados (cansat/casualties.py): densidad poblacional × área relevada × ocupación × fracción dañada, y luego × colapso × letalidad. Se publica con banda ×0.5-×2 y los supuestos van en la telemetría. NO es una predicción de víctimas y así debe presentarse.
- El protocolo LB135 v2 lleva lat/lon como campos opcionales al final (junto a danado_pct), así un receptor viejo los ignora. La Pi recibe la posición de la ESP32 por UART y la asocia a cada frame; la estación dibuja la trayectoria sin basemap (offline).
- En la Pi Zero v1 (ARMv6) no hay ultralytics/torch: la detección de personas va por el NPU del AI Camera (IMX500, --det-backend imx500) o se apaga (--no-detect). En PC/post-vuelo sigue YOLOv8s@1280. El modo IMX500 está implementado contra la API oficial de picamera2 y PENDIENTE de validación en hardware.
- El siamés queda DESACTIVADO por defecto en vuelo (--siamese-onnx para activarlo). Los pesos viejos alucinaban daño sin cambio (5.5 % medio, máx 63.6 %) y contaminaban el consenso (27.1 % de daño falso en el frame 4191 de la telemetría guardada). Re-entrenado con pares negativos bajó a 1.23 % medio / 6.5 % máx; aun así el "pre" de vuelo es un tile nadir contra frames oblicuos: cambio de dominio fuerte.
- prepare_floodnet.py remapeaba con un `shift` heurístico que desplazaba TODAS las clases: "flood" era Building-NON-flooded y "agua normal" era Tree. Corregido contra el class_mapping.csv oficial (flood = clases 1 y 3; agua = 5) y el especialista fue re-entrenado (IoU flood 0.483). Es el primer modelo del repo que pasa la auditoría IMX500.
- Opset 17, entrada fija [1,3,320,320], salida 'logits'.
- El CSV de telemetría creció de 23 a 29 columnas con las extensiones del DPD, SIEMPRE al final para no romper consumidores viejos (DictReader): lat, lon, hum_pct, area_m2, personas_afectadas, perdidas_est. La humedad cierra el gap del BME280 (el DPD la mide y no viajaba); protocol.py la lleva como campo 23 opcional del paquete v2. La estación (web_server.CSV_COLUMNS + types.ts Frame) las conoce y degrada elegantemente si faltan (CSV viejo).
- Umbral de voto del consenso de daño: 10 % (DAMAGE_CONSENSUS_PCT). CALIBRADO el 2026-09-17 con tools/calibrate_thresholds.py sobre xBD con split por desastre (120 tiles de hurricane-matthew + palu-tsunami, nunca vistos): el óptimo medido es 11.8 % (F1 0.77, FPR 4.3 %). El 10 % del repo queda dentro del margen y se mantiene como default; el pipeline lo expone como --damage-threshold para recalibrar con frames del predio. Frames sin desastre: daño mediano 1.1 %, p95 10.7 %. Con desastre: mediana 22.8 %, p95 49.3 %.
- Detector de personas: fine-tune de VisDrone (models/yolov8n_visdrone.pt) con conf=0.25 e imgsz=1280. CAMBIO 2026-09-17: antes era yolov8s COCO. Medición sobre dataset/pruebas/PERSONAS_AUTOS.png @1280: VisDrone 254 personas vs COCO 42 — los modelos COCO pierden personas sub-píxel, que es el caso del CanSat (a 250 m, 1 persona ≈ 6 px). Los IDs de clase se resuelven desde model.names en runtime (presets COCO/VisDrone). En la Pi Zero v1 la detección va por el NPU del AI Camera (--det-backend imx500, SSD COCO on-sensor): misma limitación de personas chicas, PENDIENTE de medir en la placa.
- **[PENDIENTE]** Cuantización INT8: decisión PENDIENTE DE RE-MEDICIÓN. La anterior ("descartada para vuelo, coincidencia 87.6% < umbral 95%") se basó en una medición inválida; ver modelo_vuelo.nota_precision. Se vuela en FP32 hasta re-medir. Adicionalmente, quantize_all.py usa quantize_dynamic, que en una CNN sólo cuantiza pesos: el "~2x más rápido" que prometía su docstring no es realista, y los artefactos ni cargan en cv2.dnn.
- Veredictos ambientales unificados en cansat/indices.py. Regla actual: 1. SIN DATOS            si menos del 5% de los píxeles son válidos 2. SUELO EXPUESTO       si GVI < -0.5 y veg < 10% y USI <= 1 y agua < 25% 3. ALTO ESTRÉS URBANO   si USI > 3 4. ESTRÉS MODERADO      si USI > 1 5. ZONA SALUDABLE       en otro caso Los umbrales de USI no cambiaron (3.0 / 1.0). Sí cambió el orden: antes SUELO EXPUESTO se evaluaba al final y era prácticamente inalcanzable, porque un terreno pelado con 2% de edificios ya daba USI > 1 y caía en ESTRÉS MODERADO. Se agregaron dos guards que el código original sólo tenía en su comentario ("sin verde y sin ciudad"): el de ciudad (USI <= 1) y el de agua (< 25%), porque GVI no distingue un lago de tierra desnuda.
- Los porcentajes de terreno ignoran píxeles sin datos. Desde la auditoría, "sin datos" es sólo el negro CONTIGUO AL BORDE (cansat/nodata.py border_mask), no cualquier píxel oscuro: el umbral global de luminancia descartaba sombras, asfalto oscuro y agua profunda de los frames de cámara real, y sobre ese remanente se calculaban porcentajes, USI, veredicto, sampler y alerta. Con --camera el umbral pasa a 0 automáticamente.
- Perfil de descenso del simulador alineado al DPD: eyección ~250 m, descenso 2-4 m/s.
- Lo que se llamaba "NDVI" se renombra GVI (Greenness / Verdor Index). No hay banda NIR en una cámara RGB, así que no es NDVI, y presentarlo como tal ante CONAE era un riesgo de credibilidad evitable. La columna de telemetry.csv sigue siendo `ndvi` para no romper el contrato publicado; el campo canónico es `gvi` y ambos llevan el mismo valor. El glosario del frontend (web-app/src/lib/vocab.ts) se actualizó en consecuencia.
- El diagnóstico de daño es mayoría simple de los modelos disponibles sobre 10% de superficie dañada, más una red de seguridad si dos modelos ven >25%. Antes la condición era `pct_dan > 10 and votos >= 2` con `votos` incluyendo a `pct_dan > 10`, o sea que el modelo principal tenía veto: si decía 9% y los otros dos 80%, no había alerta. Y la UI anunciaba "consenso de 3 modelos".
- Aptitud de vuelo MEDIDA con corrupción sintética (tools/stress_suite.py + cansat/corrupt.py; 120 imgs/terreno-daño, 80 flood, 60 fuego; seed 42, severidad fija declarada en el módulo; JSON: outputs/stress_suite.json). mIoU limpio → estresado: · Terreno (LoveDA Val, cansat_seg_terrain_v2): 51.0 % → lluvia 20.2 %, niebla 16.3 %, motion_blur 43.1 %, subexp 44.9 %, sobreexp 29.3 %, escala 50.2 %. · Daño xBD held-out (Joplin/Nepal, cansat_damage3): IoU bin 0.142 → lluvia 0.073, niebla 0.069, motion 0.091, subexp 0.121, sobreexp 0.069, escala 0.159. · Daño two-stage de vuelo (RescueNet val, cansat_damage_v3_bal): IoU bin 0.403 → lluvia 0.334, niebla 0.198, motion 0.350, subexp 0.356, sobreexp 0.278, escala 0.374. · Flood (FloodNet val 80 imgs, especialista 224): IoU inundación 0.489 → lluvia 0.256, niebla 0.319, motion 0.488, subexp 0.418, sobreexp 0.348, escala 0.493. · Fuego (fire_smoke valid+test 60 imgs; el valid seleccionó el checkpoint, se declara): IoU fuego 0.801 → lluvia 0.306, niebla 0.787, motion 0.706, subexp 0.642, sobreexp 0.595, escala 0.761. Lectura honesta: niebla y lluvia son las degradaciones dominantes en terreno y daño (la niebla más que duplica el error del two-stage); en flood manda la lluvia (−0.233) y en fuego la lluvia es catastrófica (−0.495) mientras la niebla casi no afecta (humo ≈ niebla para el modelo). Las exposiciones afectan moderado; el jitter de escala casi no afecta porque el resize al input normaliza. Son corrupciones SINTÉTICAS: miden robustez relativa del mismo modelo contra sí mismo, no reemplazan validación de vuelo real.
- Consulta Terrestre en la estación: consultas espaciales simbólicas sobre las máscaras de clase por frame. El post-vuelo ahora las persiste en entrega/masks/ (PNG gris de índices, 255 = sin dato; se apaga con --no-masks y suma el bucket del contrato v3). Motor: cansat/consultas.py (plantillas en español e inglés → área, conteo con buffer métrico, fracción de longitud por esqueleto, distancia, existencia y personas). NO hay LLM como respondedor: lo que no mapea devuelve "consulta no soportada" con sugerencias. El servidor de la estación lo invoca por subproceso (tools/consulta.py), así sigue siendo stdlib-only. Georreferenciado declarado APROXIMADO: cuadrado de lado sqrt(area_m2) centrado en (lat, lon), norte arriba, sin heading (la telemetría no trae IMU). La longitud es ≈ ±10 % por esqueletización. v2 (2026-09-21): el pipeline persiste las DETECCIONES por frame en telemetry.jsonl (tipo + xyxy en píxeles nativos). Con eso el motor responde conteo de personas por posición (dentro de la zona dibujada), personas a ≤ r metros de un objeto y distancia mínima/media persona→objeto. Convención: punto de apoyo = centro-abajo del bbox (sin pose ni orientación, declarado en la respuesta). Si la misión no trae detecciones, el conteo cae al total por frame del CSV y las consultas con posición se rechazan explícitamente; la demo sintética las genera. El radio del buffer se acota a la diagonal del frame: una consulta de "200 m" sobre frames de pocos metros devuelve el frame completo, no un kernel gigante (bug real detectado con la demo y corregido). Las consultas de vías tienen especialista propio (train_road_specialist.py, 2 clases sobre FloodNet road, split 80/20 seed 42 idéntico al del flood). Gate acordado: IoU de vía en FloodNet val ≥ 0.50. Resultado: FloodNet-only PASA con 0.515; la variante mixta FloodNet+LoveDA daba 0.489 (no pasaba) y 0.532 en LoveDA, o sea que mezclar dominios empeoraba el dominio UAV de medición sin ganar lo suficiente, así que se integra la FloodNet-only. ⚠ Cross-domain LoveDA medido: 0.169 — los números de vía sobre frames satelitales/oblicuos son indicativos; el dominio válido es UAV/inundación. El modelo NO vuela: corre en post_flight (--vias-onnx) y su máscara va a entrega/masks/<src>_vias.png (contrato v3).
- MIL para el tipo de desastre (bolsa = evento, instancias = tiles, pooling con atención gatada) probado contra el clasificador por tile con el MISMO LOEO (24 eventos, seed 42): media ponderada de accuracy por bolsa 0.3277 vs 0.3681 del baseline por tile. NO lo supera: se documenta como técnica que no funcionó y el clasificador por tile queda. Caveat registrado: la métrica de bolsa (accuracy por bolsa, ponderada por bolsas) no es idéntica a la de tile (ponderada por tiles); 6 épocas y K=8 con eventos chicos de alta varianza. JSON: outputs/mil_loeo.json. Per-eventos donde sí gana: moore-tornado, socal-fire, woolsey-fire y palu-tsunami (acc bolsa 0.95-1.0).
- --no-damage apaga de verdad los tres modelos de daño + siamés + flood. Antes --no-detect sólo apagaba YOLO y todo lo demás corría igual por frame. Además el flood specialist se ejecutaba DOS veces por frame (una en la rama de diagnóstico y otra en el bloque del sampler) y el tensor del baseline siamés se releía y reprocesaba en cada frame aunque la imagen no cambia. Los tres eran cuellos de botella en la Pi Zero 2 W.

## 8. Limitaciones conocidas del modelo
- Agua turbia/barrosa puede clasificarse como bare_ground (LoveDA tiene agua azulada).
- Senderos angostos (< ~1 px a 320x320) se absorben en la clase dominante del entorno.
- Detección de personas válida sólo a baja altitud; en tiles satelitales las personas son sub-píxel. Los conteos de los simulacros hechos sobre LoveDA Test NO son significativos y no deberían citarse como resultado.
- El modelo "alucina" clases sobre regiones sin datos; se excluyen con máscara de validez. Con la máscara nueva (sólo borde) un frame entero oscuro produce el veredicto "SIN DATOS" en vez de "ZONA SALUDABLE", que es lo que daba antes.
- La clase 'other' agrupa calles y estructuras; no existe clase 'road' propia.
- USI no está acotado a [0,1]: es el cociente bui/veg con un piso de 0.1 en el denominador, así que puede valer cientos (bui=50%, veg~0 → USI=500). Para mostrar una barra 0-1 usar usi_norm, que satura en USI=300. El tooltip del frontend decía "0 = natural, 1 = todo construido", lo cual era falso; se corrigió.
- GVI no distingue agua de suelo desnudo (ambos dan ~-1 cuando veg=0). Por eso el veredicto SUELO EXPUESTO exige además agua < 25%: un lago ya no sale como "suelo expuesto". Tampoco hay un veredicto propio para agua; cae en ZONA SALUDABLE, que es discutible y conviene revisar si el jurado lo pregunta.
- Benchmark del parser contra los QA reales de EarthVQA (Val, 57 202 preguntas = 51 distintas; dataset gated, QA local en dataset/earthvqa/, uso académico no comercial). Herramienta: tools/bench_parser_earthvqa.py. Cobertura: 41.4 % global; Basic Counting 85.7 % y Basic Judging 85.7 % (falta solo 'playground', que no existe en la ontología de 5 clases); Reasoning-based Judging 16.3 % (agricultural land/woodland/uncultered → vegetación/vegetación/suelo); Comprehensive, Object Situation y Reasoning Counting 0 % (tipos, usos de suelo, intersecciones, eutrófico: fuera de alcance, se rechazan). Precisión de mapeo 100 % (15/15) y de rechazo 100 % (36/36) sobre el inventario COMPLETO de 51 preguntas distintas, etiquetado a mano. Dos falsos soportes detectados a escala y corregidos antes de medir: 'uncultivated agricultural land' (es barbecho → suelo, no vegetación) y 'construction land/area' (uso de suelo: se rechaza). Aproximaciones declaradas: agriculture/woodland→vegetación y uncultivated→suelo; 'roads' usa el especialista de vías (cross-domain LoveDA 0.169). NO mide exactitud de respuesta VQA. Evidencia versionada: docs/benchmarks/earthvqa_parser.json y la muestra adjudicada a mano docs/benchmarks/earthvqa_parser_muestra_51.csv.
- El baseline siamés (outputs/baseline.png) es un tile satelital nadir de Esri World Imagery a zoom 18 (~0.6 m/px, 640x640 → ~380 m de lado), y los frames de vuelo son oblicuos desde ~250 m con otro FOV y otra resolución. El cambio de dominio hace que el siamés tienda a marcar "cambio" en todas partes. Tratar su salida como señal débil, no como prueba.
- Los modelos de daño se entrenaron con xBD (CC BY-NC-SA 4.0), que es atribución + no comercial + compartir igual. Los pesos derivados heredan esa licencia. LoveDA en cambio es CC BY 4.0 (sin restricción comercial). Ver la sección "Datasets y licencias" del README antes de distribuir nada.
- train_siamese_damage.py tenía un fallback `pre, post = img, img` cuando faltaba el par pre/post, o sea que entrenaba a la red a predecir daño donde NO hay ningún cambio. Corregido (ahora skipea la muestra), pero los pesos best_siamese_damage.pth ya entrenados arrastran ese defecto: re-entrenar antes de confiar en el siamés.

## 9. Próximos pasos
- Convertir el ONNX a formato IMX500 al llegar la Raspberry Pi, con las herramientas oficiales. pi/convertir_modelos.sh NO fue probado (lo admite en un comentario): pegar la salida de --help y ajustar la sintaxis.
- Preparar la microSD según la placa REAL: · Pi Zero W v1 (ARMv6): Raspberry Pi OS 32-bit + AI Camera + backend cv2.dnn (no hay wheels de onnxruntime) + modelos autocontenidos (tools/onnx_inline.py). Guía paso a paso en pi/guia_pi.md. · Pi Zero 2 W / Pi 4 / Pi 5: Raspberry Pi OS 64-bit + requirements-flight.txt. Medir s/frame en la placa antes de fijar --frames/--interval de vuelo.
- Verificar requirements-flight.txt EN UNA PI FÍSICA. pi/guia_pi.md instalaba numpy + opencv-python-headless + onnxruntime y omitía ultralytics, que mission_pipeline.py importa incondicionalmente: el script de vuelo no arrancaba. Verificar además que el flavour de OpenCV incluya dnn_superres si se va a correr EDSR en la Pi.
- Re-medir la validación INT8 con validate_int8_mission.py corregido y recién ahí decidir FP32 vs INT8. Actualizar modelo_vuelo.nota_precision y la decisión `int8` con el resultado.
- Prueba real de UART Pi <-> ESP32 con el protocolo unificado de cansat/protocol.py. Los tres formatos anteriores eran incompatibles entre sí (el listener no podía leer lo que emitía el pipeline), así que esta prueba nunca se pudo hacer.
- Ejecutar CHECKLIST-SIMULACRO.md de la estación terrena con sus cuatro variantes obligatorias y llenar la tabla "Registro de corridas", que está vacía. La fila 7 exige "cero errores de consola": verificarlo con el filtro de consola del frontend desactivado, porque parchea console.warn.
- Regenerar las métricas con evaluate.py (unificado) y citar en el DPD sólo números que salgan de outputs/metrics/*.json. En el repo convivían cuatro mIoU distintos para el mismo modelo (46.36%, 52.44%, 0.5317 y "~0.5317"), tres de ellos hardcodeados en un print().
