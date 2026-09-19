"""
CanSat La Base — Pipeline de misión (v7).

Captura → segmentación + índices ambientales + consenso de daño + detección
        → telemetría por radio/UART + frames de evidencia anotados.

Formato del paquete de radio: ver ``cansat/protocol.py`` (contrato v2).
Columnas de ``telemetry.csv``: ver ``CSV_COLUMNS`` más abajo y ``README.md``.

Uso en PC (prueba con tiles):
    python mission_pipeline.py --folder dataset/loveda_raw/Test/Urban/images_png --frames 4

Sin imágenes de evidencia (más rápido):
    python mission_pipeline.py --folder ... --no-vis

Uso en Raspberry (vuelo):
    python mission_pipeline.py --camera --frames 60 --interval 2 --p0-alt 20

────────────────────────────────────────────────────────────────────────────
CAMBIOS DE v7 (auditoría 2026-09-16)
────────────────────────────────────────────────────────────────────────────
Rendimiento en la Pi (era el bloqueante principal):
  · ``--no-damage`` apaga de verdad los 3 modelos de daño + siamés + flood.
    Antes ``--no-detect`` sólo apagaba YOLO y todo lo demás corría igual.
  · El flood specialist se ejecuta UNA vez por frame (antes: dos).
  · El tensor del baseline siamés se cachea fuera del loop (antes se leía y
    preprocesaba ``outputs/baseline.png`` en cada frame).
  · Los ``print`` de configuración salieron del loop de frames.

Robustez de vuelo:
  · ``bmp_read()`` con try/except y fallback a atmósfera simulada. Antes una
    falla de I2C en pleno descenso mataba el pipeline y se perdía la misión.
  · ``--p0-alt`` permite calibrar la presión de referencia con la altitud
    conocida del predio. Antes ``p0 = bmp.pressure`` usaba la presión local,
    así que la altitud arrancaba en ~0 m y derivaba con el clima.
  · Los ONNX obligatorios se validan al arranque con mensaje accionable; los
    opcionales avisan por warning en vez de desactivarse en silencio.
  · Telemetría en modo append con sufijo horario (``--overwrite`` para el
    comportamiento viejo). Antes cada corrida borraba el vuelo anterior.

Correcciones de lógica:
  · El consenso de daño es un consenso de verdad (mayoría de modelos), sin el
    veto del modelo principal. Ver ``cansat.indices.diagnose``.
  · Índices y veredictos unificados en ``cansat.indices`` (antes había tres
    definiciones de USI/NDVI con umbrales distintos).
  · Incertidumbre por entropía del softmax en la pasada única, así que el
    ``AdaptiveSampler`` tiene señal sin necesitar ``--tta`` (4× inferencia).
    Antes ``uncert`` era 0.0 salvo con ``--tta``, y el término de incertidumbre
    del score y la bandera de "revisión humana" estaban inertes.
  · ``--crf`` devuelve log-probabilidades, no probabilidades: ya no se mezclan
    espacios al combinarlo con ``--temporal``. La implementación es única
    (``cansat.crf.dense_crf``, filtro guiado por la imagen) y ya no cae a un
    bilateral ciego cuando falta opencv-contrib.
  · ``--sharp-gate`` marca los frames borrosos y ``--sharp-drop`` los descarta
    de verdad (sin inferencia, sin telemetría). Un frame marcado ya no contamina
    el suavizado temporal.
  · La incertidumbre por entropía y la varianza de TTA están **en la misma
    escala [0,1]**, así que el umbral ``--uncert-max`` del sampler aplica a
    ambas. Antes la varianza de TTA (0.001-0.01) se comparaba contra 0.03 fijo y
    la entropía (0.3-0.7) contra el mismo 0.03: el sampler veía "todo incierto".
  · ``aff_m2`` usa la fórmula correcta de huella en tierra: ``2·alt·tan(FOV/2)``.
    Antes faltaba el ``/2`` (FOV doble) y el área salía ×4.6.
  · ``--nodata-mode border`` descarta sólo el negro contiguo al borde, así las
    sombras y el asfalto oscuro de un frame de cámara real no se pierden como
    "sin datos". Con ``--camera`` el umbral pasa a 0 automáticamente.
  · Nombres de nodo ONNX leídos del modelo, no hardcodeados. El mapeo de
    entradas del siamés ahora es por nombre y avisa si cae a posicional.
  · Con ``--no-damage`` el consenso recibe ``None`` (modelos desactivados no
    votan). Antes recibía ``0.0`` y "votaban" en contra.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# ── OpenCV con mensaje accionable (es la dependencia que falta en la Pi) ── #
try:
    import cv2
except ImportError as _e:                                     # pragma: no cover
    sys.stderr.write(
        "[ERROR] Falta OpenCV.\n"
        "  PC        : pip install opencv-python\n"
        "  Raspberry : pip install opencv-python-headless\n"
        "  Si vas a usar super-resolución (post-vuelo) necesitás el flavour\n"
        "  'contrib': pip install opencv-contrib-python-headless\n"
    )
    raise SystemExit(1) from _e

# onnxruntime es OPCIONAL: si no está instalado (p. ej. Pi Zero v1 / ARMv6, que
# no tiene wheels), ``cansat.onnxio`` cae al backend ``cv2.dnn``, que ya viene
# con OpenCV. Antes esto era un ``SystemExit`` y el script no arrancaba en esa
# placa. El backend efectivo se imprime por modelo en el banner.
try:
    import onnxruntime as ort  # noqa: F401
    _ORT_DISPONIBLE = True
except ImportError:                                           # pragma: no cover
    _ORT_DISPONIBLE = False

# ── Biblioteca compartida (fuente única de índices, protocolo y rutas) ──── #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat import indices as IDX
from cansat import nodata as ND
from cansat import onnxio
from cansat import preprocess as PP
from cansat import population as POP
from cansat import protocol as PROTO
from cansat import stress as ST
from cansat.casualties import Supuestos, estimar as estimar_perdidas
from cansat.crf import dense_crf

from adaptive_sampler import AdaptiveSampler
from inference import create_colored_mask

# ── Reexport por compatibilidad con scripts que importaban de acá ──────── #
CLASS_NAMES = list(IDX.CLASS_NAMES)
MEAN, STD = PP.MEAN, PP.STD
VERDICT_CODE = IDX.VERDICT_CODE

# Columnas de telemetry.csv — CONTRATO con web_server.py y el frontend.
# Si se agrega una columna, actualizar también CSV_COLUMNS en web_server.py y
# Frame en web-app/src/lib/types.ts.
CSV_COLUMNS = [
    "t_s", "alt_m", "p_hpa", "temp_c",
    "veg", "bui", "wat", "bare", "oth", "dom",
    "usi", "ndvi", "verdict",
    "people", "vehicles", "danado_pct", "aff_m2", "diag", "alert",
    "sharp", "src", "sample_pri", "sample_score",
    # ── Extensiones DPD (agregadas al final: los consumidores viejos que leen
    #    por DictReader siguen funcionando) ──────────────────────────────────
    "lat", "lon",           # posición GPS (0/0 = sin fix)
    "hum_pct",              # humedad relativa del BME280 (0 = sin dato)
    "area_m2",              # huella en tierra del frame
    "personas_afectadas",   # estimación de exposición (cansat.casualties)
    "perdidas_est",         # estimación de pérdidas humanas (con supuestos)
    "fire_pct",             # fuego detectado (F3; vacío si el modelo no está)
    "smoke_pct",            # humo detectado (F3)
    "haze_pct",             # bruma/aerosoles por dark channel (estrés ambiental)
    "humidex",              # calor con T y humedad del sensor (estrés térmico)
    "stress_idx",           # índice agregado 0-100 (bruma 50 % + calor 25 % + urbano 25 %)
    "colapso_pct",          # fracción de colapso MEDIDA (severidad; vacío = supuesto)
]

# IDs de clase COCO. Se validan contra ``model.names`` en runtime: antes estaban
# duplicados (constante del módulo + string del CLI) y la constante no la usaba
# nadie, así que leer el código daba una idea equivocada de lo que corría.
COCO_PERSON = [0]
COCO_VEHICLES = [1, 2, 3, 5, 7]        # bicycle, car, motorcycle, bus, truck
VISDRONE_PERSON = [0, 1]               # pedestrian, people
VISDRONE_VEHICLES = [2, 3, 4, 5, 6, 7, 8]

# FOV de la cámara (grados). Usado para estimar el área en tierra por frame.
FOV_H_DEG = 33.0
FOV_V_DEG = 26.15

# Escala de saturación de la incertidumbre de TTA (varianza media de las 4
# vistas). Junto con entropy_uncertainty() deja las dos fuentes de
# incertidumbre en [0,1] para que --uncert-max aplique a ambas.
TTA_UNCERT_REF = 0.05

# Lado máximo de la copia donde se calcula la bruma (dark channel): en la Pi
# Zero v1 el erode a resolución completa costaría segundos por frame.
STRESS_MAX_SIDE = 320.0


def _hash8(path: str | Path) -> str:
    """
    sha256 corto (primeros 4 MB) de un ONNX: identifica el artefacto exacto.

    Trazabilidad F6: cada frame del JSONL registra con qué modelo (hash) y en
    qué precisión se infirió, para no citar métricas de otro artefacto.
    """
    p = Path(path)
    if not p.is_file():
        return "?"
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(4 << 20))
    return h.hexdigest()[:8]


def ground_area_m2(alt_m, fov_h_deg=FOV_H_DEG, fov_v_deg=FOV_V_DEG):
    """
    Huella en tierra del frame a una altitud dada, en m².

    ⚠ FIX: la versión anterior era ``2·alt·tan(FOV)`` — la fórmula de un FOV
      DOBLE, no del semiángulo. Sobreestimaba el área por ``tan(33°)/tan(16.5°)``
      ≈ 2.2 por lado (×4.6 el área) y con eso el ``aff_m2`` del DPD. La huella
      de una cámara es ``2·alt·tan(FOV/2)`` por lado.
    """
    w_m = 2.0 * float(alt_m) * math.tan(math.radians(fov_h_deg / 2.0))
    h_m = 2.0 * float(alt_m) * math.tan(math.radians(fov_v_deg / 2.0))
    return max(0.0, w_m) * max(0.0, h_m)


# ══════════════════════════════════════════════════════════════════════ #
#  Hardware
# ══════════════════════════════════════════════════════════════════════ #
def detect_hardware():
    """
    Detecta cámara y sensor barométrico.

    Devuelve ``(cam_ok, sensor)`` con ``sensor`` en ``{"bme280", "bmp280", None}``.
    El DPD usa **BME280** (presión, temperatura y humedad); si solo hay BMP280
    (sin humedad) el pipeline funciona igual y la humedad queda en 0.
    """
    cam_ok = importlib.util.find_spec("picamera2") is not None
    if importlib.util.find_spec("board") is None:
        return cam_ok, None
    if importlib.util.find_spec("adafruit_bme280") is not None:
        return cam_ok, "bme280"
    if importlib.util.find_spec("adafruit_bmp280") is not None:
        return cam_ok, "bmp280"
    return cam_ok, None


def bmp_read(sensor, p0):
    """
    Lectura barométrica **a prueba de fallos**.

    El I2C de la Pi se puede colgar por vibración o por un cable que se mueve.
    Antes esta función no tenía try/except (sí lo tenía la inicialización), así
    que una falla en pleno descenso mataba el pipeline y se perdía toda la
    misión. Ahora devuelve ``None`` y el llamador degrada a atmósfera simulada.

    Devuelve ``(alt, p, temp, hum)`` con ``hum=None`` si el sensor no mide
    humedad (BMP280).
    """
    try:
        pres = float(sensor.pressure)
        temp = float(sensor.temperature)
        hum = float(sensor.humidity) if hasattr(sensor, "humidity") else None
    except Exception as e:
        return None, f"sensor: {type(e).__name__}: {e}"
    if not (300.0 < pres < 1200.0):
        return None, f"sensor: presión fuera de rango ({pres:.1f} hPa)"
    if hum is not None and not (0.0 <= hum <= 100.0):
        hum = None
    return (PROTO.pressure_to_altitude(pres, p0), pres, temp, hum), None


# ══════════════════════════════════════════════════════════════════════ #
#  Pre/pos-procesado de imagen
# ══════════════════════════════════════════════════════════════════════ #
def preprocess_bgr(bgr, img_size):
    """BGR uint8 → NCHW float32. Delega en ``cansat.preprocess`` (antes duplicado)."""
    return PP.preprocess_bgr(bgr, img_size)


def enhance_frame(bgr):
    """Denoise + unsharp suave: combo ganador del bench v2 (+5.6 pts con borrón)."""
    den = cv2.bilateralFilter(bgr, 7, 50, 50)
    g = cv2.GaussianBlur(den, (0, 0), 3)
    return cv2.addWeighted(den, 1.5, g, -0.5, 0)


def sharp_score(bgr):
    """Nitidez por varianza del Laplaciano. Usada por ``--sharp-gate``."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def softmax_np(x):
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def entropy_uncertainty(logits, valid=None):
    """
    Incertidumbre barata: entropía media del softmax, normalizada a [0, 1].

    Reemplaza la dependencia de ``--tta`` (4× inferencia) para tener señal de
    incertidumbre en la pasada única.

    ``valid`` (H,W bool) excluye los píxeles sin datos del promedio. Sin esto,
    un tile con 50 % de borde negro daba ``uncert`` inflado (~0.5) y el sampler
    marcaba para revisión frames que en realidad eran seguros: el borde negro
    no es "ambigüedad del modelo".
    """
    pr = softmax_np(logits[0])
    pr = np.clip(pr, 1e-9, 1.0)
    H = -(pr * np.log(pr)).sum(axis=0)
    denom = math.log(max(2, pr.shape[0]))
    if valid is not None and valid.shape == H.shape and valid.any():
        return float(H[valid].mean() / denom)
    return float(H.mean() / denom)


def tta_logits(sess, bgr, img_size):
    """
    TTA: promedio de logits sobre 4 vistas + incertidumbre en [0,1].

    La incertidumbre es la varianza media de las probabilidades entre vistas,
    normalizada con ``TTA_UNCERT_REF`` para que comparta escala con
    ``entropy_uncertainty`` (que también devuelve [0,1]).
    """
    acc = None
    probs = []
    for img, flip in [(bgr, None), (cv2.flip(bgr, 1), "h"),
                      (cv2.flip(bgr, 0), "v"), (cv2.flip(bgr, -1), "hv")]:
        lg = sess.run({"input": PP.preprocess_bgr(img, img_size)})
        if flip == "h":
            lg = lg[:, :, :, ::-1]
        elif flip == "v":
            lg = lg[:, :, ::-1, :]
        elif flip == "hv":
            lg = lg[:, :, ::-1, ::-1]
        probs.append(softmax_np(lg[0]))
        acc = lg if acc is None else acc + lg
    var = float(np.mean(np.var(np.stack(probs), axis=0)))
    return acc / 4.0, min(1.0, var / TTA_UNCERT_REF)


def crf_lite(logits, bgr, iters=3):
    """
    CRF-lite: paso de mensajes con filtro guiado por la imagen.

    ⚠ UNIFICADO: antes había dos implementaciones que no se parecían — el
      guided filter de acá y el mean-field de ``cansat/crf.py`` — y además la de
      acá caía a un ``bilateralFilter`` sobre las propias probabilidades cuando
      faltaba opencv-contrib, sin usar la imagen. Ahora delega en
      ``cansat.crf.dense_crf``, que guía con la imagen y usa box filters core
      (funciona en la Pi aunque no esté contrib).

    ⚠ Devuelve **log-probabilidades**, no probabilidades. La versión anterior
      devolvía ``probs`` y el llamador las asignaba a ``logits``; al combinar
      ``--crf`` con ``--temporal`` se promediaban probabilidades [0,1] con
      logits sin acotar, y ``terrain_percentages`` hacía argmax sobre un espacio
      mixto. Con log-probs el espacio es consistente en todo el pipeline.
    """
    probs = softmax_np(logits[0])
    refined = dense_crf(probs, bgr, iters=iters)
    return np.log(np.clip(refined, 1e-9, 1.0))[np.newaxis, ...]


def valid_mask_for(bgr, img_size, thresh=15, mode="border"):
    """Máscara booleana (img_size, img_size) de píxeles con datos."""
    if thresh <= 0:
        return np.ones((img_size, img_size), dtype=bool)
    mask = (ND.border_mask(bgr, thresh) if mode == "border"
            else ND.threshold_mask(bgr, thresh))
    return ND.valid_at_size(mask, img_size)


def terrain_percentages(logits, bgr, img_size, thresh=15, mode="border", valid=None):
    """Porcentajes por clase sobre píxeles válidos. Ver ``cansat.nodata``."""
    seg = np.argmax(logits[0], axis=0)
    if valid is None:
        valid = valid_mask_for(bgr, img_size, thresh, mode)
    pcts, dom, _n = ND.terrain_percentages(seg, valid, len(IDX.CLASS_NAMES))
    return pcts, dom, seg, valid


def _rnum(v, nd=1):
    """``round`` que tolera ``None`` (modelo de daño desactivado → no votó)."""
    return None if v is None else round(v, nd)


def _disp(v, nd=1):
    """Formato de consola para un valor de daño que puede ser ``None``."""
    return "off" if v is None else f"{v:.{nd}f}"


# ══════════════════════════════════════════════════════════════════════ #
#  Detección
# ══════════════════════════════════════════════════════════════════════ #
def load_yolo(model_path, person_ids, veh_ids):
    """
    Carga YOLO y **valida los IDs de clase contra ``model.names``**.

    Antes los IDs vivían en dos lugares (constantes del módulo y string del CLI)
    y la constante no la usaba nadie. Si alguien pasaba IDs de VisDrone con
    pesos COCO, contaba clases equivocadas sin avisar.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.stderr.write(
            "[ERROR] Falta ultralytics (pip install ultralytics).\n"
            "  Es la dependencia que pi/guia_pi.md NO instalaba en la Pi, así\n"
            "  que el script de vuelo no arrancaba. Ver requirements-flight.txt.\n"
            "  Si no la necesitás, pasá --no-detect.\n"
        )
        raise SystemExit(1) from None

    if not Path(model_path).is_file():
        sys.stderr.write(
            f"[ERROR] No existe el detector: {model_path}\n"
            f"  En el repo hay models/yolov8s.pt. Ultralytics descargaría al\n"
            f"  CWD en runtime, que en vuelo (sin red) falla: usá la ruta local.\n"
        )
        raise SystemExit(1)

    yolo = YOLO(model_path)
    names = getattr(yolo, "names", {}) or {}
    if names:
        bad = [(i, names.get(i, "?")) for i in sorted(person_ids | veh_ids)
               if i in names and names[i] not in
               ("person", "pedestrian", "people", "bicycle", "car", "motorcycle",
                "bus", "truck", "van", "caravan", "motorbike", "cyclist")]
        if bad:
            print(f"  [WARN] IDs de clase sospechosos para {Path(model_path).name}: "
                  + ", ".join(f"{i}='{n}'" for i, n in bad))
    return yolo


def detect_objects(yolo, bgr, conf, imgsz, person_ids, veh_ids):
    """Cuenta personas/vehículos y devuelve también los recuadros."""
    res = yolo.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)
    people, veh = 0, 0
    boxes = []
    for box in res[0].boxes:
        c = int(box.cls[0])
        if c in person_ids:
            people += 1
        elif c in veh_ids:
            veh += 1
        else:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2, c))
    return people, veh, boxes


# ══════════════════════════════════════════════════════════════════════ #
#  Evidencia visual
# ══════════════════════════════════════════════════════════════════════ #
def build_evidence(bgr, seg, boxes, env, people, veh, alt, p, temp, t,
                   nodata_pct=0.0, nodata_thresh=15, person_ids=frozenset({0}),
                   diag=None, sharp=None, sharp_ok=True):
    """Imagen de evidencia: overlay + recuadros + cartel."""
    h, w = bgr.shape[:2]
    mask = create_colored_mask(seg)
    mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Mezcla solo donde hay datos; el negro de borde queda negro.
    blended = cv2.addWeighted(bgr, 0.6, mask_r, 0.4, 0)
    if nodata_thresh > 0:
        valid3 = (bgr.max(axis=2) > nodata_thresh)[..., np.newaxis]
        vis = np.where(valid3, blended, bgr).astype(np.uint8)
    else:
        vis = blended

    for (x1, y1, x2, y2, c) in boxes:
        color = (0, 255, 0) if c in person_ids else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    lines = [
        f"t={t:05.1f}s  alt={alt:6.1f}m  p={p:6.1f}hPa  T={temp:4.1f}C",
        f"{env['verdict']}   USI={env['usi']:.2f}  GVI={env['gvi']:+.2f}",
        f"personas={people}  vehiculos={veh}",
    ]
    if diag:
        lines.append(f"DIAG: {diag}")
    if nodata_pct > 1.0:
        lines.append(f"SIN DATOS: {nodata_pct:.1f}% (no analizado)")
    if sharp is not None and not sharp_ok:
        lines.append(f"FRAME BORROSO (sharp {sharp:.0f}) - no usado")

    cv2.rectangle(vis, (0, 0), (w, 26 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, 22 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


# ══════════════════════════════════════════════════════════════════════ #
#  CLI
# ══════════════════════════════════════════════════════════════════════ #
def build_parser():
    ap = argparse.ArgumentParser(
        description="Pipeline de misión CanSat LB135",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_argument_group("fuente de frames")
    src.add_argument("--image", default=None, help="una sola imagen")
    src.add_argument("--folder", default=None, help="carpeta con .png")
    src.add_argument("--camera", action="store_true", help="picamera2 (Raspberry)")
    src.add_argument("--frames", type=int, default=5)
    src.add_argument("--interval", type=float, default=2.0, help="segundos entre frames")
    src.add_argument("--shuffle", action="store_true", help="muestrear al azar (simulacro)")
    src.add_argument("--seed", type=int, default=42)

    mod = ap.add_argument_group("modelos")
    mod.add_argument("--onnx", default=None, help="segmentación de terreno (default: FP32 de vuelo)")
    mod.add_argument("--img-size", type=int, default=320)
    mod.add_argument("--damage-onnx", default="outputs/cansat_damage3_mobilenetv2.onnx")
    mod.add_argument("--damage2-onnx", default="outputs/cansat_damage_v3_bal.onnx",
                     help="two-stage de vuelo: adaptado a UAV con RescueNet "
                          "(F2 2026-09-18). El xBD puro queda como alternativa: "
                          "outputs/cansat_damage_v3.onnx")
    mod.add_argument("--flood-onnx", default="outputs/cansat_flood_specialist_224.onnx",
                     help="especialista de inundación a 224 px (F3 2026-09-18, "
                          "IoU 0.489 y la mitad de cómputo que el de 320). "
                          "Alternativa: outputs/cansat_flood_specialist.onnx")
    mod.add_argument("--fire-onnx", default="outputs/cansat_fire_smoke.onnx",
                     help="detector de fuego/humo (F3 2026-09-18, media IoU "
                          "0.762). Opcional: si falta, no se usa.")
    mod.add_argument("--severity-onnx", default="outputs/cansat_severity.onnx",
                     help="severidad del daño (5 clases, F2b 2026-09-18). Si "
                          "está, la fracción de colapso es MEDIDA y reemplaza "
                          "el supuesto 0.3 de casualties. Opcional.")
    mod.add_argument("--no-stress", action="store_true",
                     help="apagar el estrés ambiental por imagen (bruma/dark "
                          "channel); el humidex usa igual los sensores")
    mod.add_argument("--siamese-onnx", default="",
                     help="siamés de cambio pre/post. DESACTIVADO por defecto: los "
                          "pesos del repo están marcados ROTO en MODELS.yaml "
                          "(alucinaban daño sin cambio y metían 27 %% de daño falso "
                          "en el consenso). Re-habilitar sólo con pesos re-entrenados "
                          "y verificados con el autochequeo de pre==post.")
    mod.add_argument("--baseline", default="outputs/baseline.png",
                     help="imagen 'pre' del cambio (la genera baseline_tool.py)")
    mod.add_argument("--damage-threshold", type=float, default=None, metavar="PCT",
                     help="umbral de voto del principal y del siamés (default: 10 %% de "
                          "cansat.indices.DAMAGE_CONSENSUS_PCT). Calibrado con "
                          "tools/calibrate_thresholds.py: óptimo medido 11.8 %%.")
    mod.add_argument("--damage-threshold-two-stage", type=float, default=None,
                     metavar="PCT",
                     help="umbral de voto del two-stage (default: 3.7 %%, calibrado). "
                          "El two-stage va enmascarado por edificios y sus valores son "
                          "~3× menores: con el umbral general casi nunca votaba.")

    det = ap.add_argument_group("detección YOLO")
    det.add_argument("--no-detect", action="store_true", help="apagar la detección")
    det.add_argument("--det-backend", choices=("auto", "yolo", "imx500", "none"),
                     default="auto",
                     help="motor de detección: yolo (ultralytics, PC/Pi de 64 bits) | "
                          "imx500 (NPU del AI Camera, on-sensor: es el camino de la "
                          "Pi Zero v1 sin torch) | none | auto (yolo si está)")
    det.add_argument("--det-model", default="models/yolov8n_visdrone.pt",
                     help="detector. Default: el fine-tune de VisDrone (aéreo). "
                          "Medido el 2026-09-17 sobre una imagen aérea de prueba: "
                          "254 personas detectadas vs 42 del yolov8s COCO (los "
                          "modelos COCO pierden personas sub-píxel en vista aérea). "
                          "Los IDs de clase se eligen solos según el nombre del modelo.")
    det.add_argument("--imx500-model",
                     default="/usr/share/imx500-models/"
                             "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
                     help="red .rpk para el NPU del AI Camera (imx500-all la instala)")
    det.add_argument("--det-conf", type=float, default=0.25)
    det.add_argument("--det-imgsz", type=int, default=1280)
    det.add_argument("--det-person", default=None,
                     help="IDs de persona. Default según el modelo: COCO 0 | VisDrone 0,1 | "
                          "IMX500 SSD 1")
    det.add_argument("--det-vehicles", default=None,
                     help="IDs de vehículo. Default: COCO 1,2,3,5,7 | VisDrone 2..8 | "
                          "IMX500 SSD 2,4,6,8")

    perf = ap.add_argument_group("costo computacional (crítico en la Pi Zero 2 W)")
    perf.add_argument("--no-damage", action="store_true",
                      help="apagar los modelos de daño/siamés/flood. NO es lo mismo que "
                           "--no-detect: antes el daño corría siempre.")
    perf.add_argument("--no-vis", action="store_true", help="no guardar imágenes de evidencia")
    perf.add_argument("--tta", action="store_true", help="test-time augmentation (4 vistas)")
    perf.add_argument("--crf", action="store_true", help="CRF-lite guiado por la imagen")
    perf.add_argument("--crf-iters", type=int, default=3,
                      help="iteraciones del refino CRF (0 = sin refino)")
    perf.add_argument("--temporal", action="store_true", help="suavizado temporal entre frames")
    perf.add_argument("--temporal-alpha", type=float, default=0.6)
    perf.add_argument("--sampler", action="store_true", help="muestreo adaptativo por prioridad")
    perf.add_argument("--uncert-max", type=float, default=0.5, metavar="U",
                      help="umbral de incertidumbre [0,1] para marcar un frame como "
                           "'revisar' (entropía normalizada o varianza de TTA). El "
                           "JSONL registra 'uncert' por frame: calibrá U con el p95 "
                           "de un vuelo de prueba.")
    perf.add_argument("--sharp-gate", type=float, default=0.0, metavar="UMBRAL",
                      help="marcar frames con Laplaciano < UMBRAL (0 = desactivado; "
                           "el bench v2 dio 50-80 para frames nítidos)")
    perf.add_argument("--sharp-drop", action="store_true",
                      help="además de marcar, DESCARTAR los frames borrosos: no se "
                           "infieren, no van a telemetría ni a evidencia")
    perf.add_argument("--cv2-threads", type=int, default=0, metavar="N",
                      help="hilos de OpenCV (0 = auto: 1 en placas de un solo "
                           "núcleo como la Pi Zero v1, default de OpenCV si no)")

    q = ap.add_argument_group("calidad de imagen / datos sin valor")
    q.add_argument("--enhance", action="store_true", help="denoise+unsharp antes de segmentar")
    q.add_argument("--nodata-thresh", type=int, default=None,
                   help="umbral de píxel 'sin datos'. Default: 15 para --folder, "
                        "0 para --camera (una cámara no produce bordes negros)")
    q.add_argument("--nodata-mode", choices=("border", "global"), default="border",
                   help="border = sólo el negro contiguo al borde (correcto para tiles); "
                        "global = umbral sobre toda la imagen (comportamiento viejo)")

    atm = ap.add_argument_group("atmósfera")
    atm.add_argument("--apogee", type=float, default=250.0, help="altitud de eyección (DPD ~250 m)")
    atm.add_argument("--speed", type=float, default=3.0, help="velocidad de descenso (DPD 2-4 m/s)")
    atm.add_argument("--p0-alt", type=float, default=None, metavar="M",
                     help="altitud conocida del punto de lanzamiento, para calibrar la "
                          "presión de referencia del BMP280 (QNH local). Sin esto la "
                          "altitud arranca en ~0 m y deriva con el clima.")
    atm.add_argument("--uart-state", default=None, metavar="JSON",
                     help="archivo de estado de uart_listener.py: si no hay BMP280, usa "
                          "su última lectura fresca (<=10 s) en vez de simular la "
                          "atmósfera, y toma la posición GPS de la ESP32.")
    atm.add_argument("--lat", type=float, default=0.0,
                     help="posición fija (demo/simulacro) si no hay GPS por UART")
    atm.add_argument("--lon", type=float, default=0.0)

    cas = ap.add_argument_group("estimación de pérdidas humanas (DPD)")
    cas.add_argument("--pop-density", type=float, default=1500.0, metavar="HAB/KM2",
                     help="densidad poblacional del área relevada. El default (1500) "
                          "es urbano-periurbano típico; ajustar con el dato del predio. "
                          "Ver cansat/casualties.py para las referencias.")
    cas.add_argument("--occupancy", type=float, default=0.6,
                     help="fracción de la población presente en el momento del vuelo")
    cas.add_argument("--collapse-frac", type=float, default=0.3,
                     help="de lo dañado, fracción que colapsó (el modelo de daño no "
                          "distingue colapso; valor conservador documentado)")
    cas.add_argument("--fatality-ratio", type=float, default=0.1,
                     help="muertes entre ocupantes de estructuras colapsadas")

    out = ap.add_argument_group("salida")
    out.add_argument("--out-dir", default="outputs/mission")
    out.add_argument("--overwrite", action="store_true",
                     help="pisar telemetry.csv en vez de abrir una corrida nueva")
    out.add_argument("--radio-format", choices=("v2", "v1", "off"), default="v2",
                     help="v2 = con número de paquete y checksum; v1 = legacy 18 campos")
    return ap


# ══════════════════════════════════════════════════════════════════════ #
#  Main
# ══════════════════════════════════════════════════════════════════════ #
def main(argv=None):
    args = build_parser().parse_args(argv)
    supuestos = Supuestos(pop_density=args.pop_density, occupancy=args.occupancy,
                          collapse_frac=args.collapse_frac, fatality=args.fatality_ratio)
    # GPS: si hay UART state se sobreescribe por frame; --lat/--lon sirven de
    # posición fija para demos/simulacros sin hardware.
    lat_f, lon_f = float(args.lat), float(args.lon)

    # ── Fuentes ─────────────────────────────────────────────────────────
    if args.image:
        frames = [(Path(args.image).stem, Path(args.image))]
    elif args.folder:
        imgs = list(Path(args.folder).glob("*.png"))
        if not imgs:
            print(f"[ERROR] No hay .png en {args.folder}")
            raise SystemExit(1)
        if args.shuffle:
            random.Random(args.seed).shuffle(imgs)
        else:
            imgs = sorted(imgs)
        frames = [(p.stem, p) for p in imgs[:args.frames]]
    elif args.camera:
        frames = [(f"cam_{i:03d}", None) for i in range(args.frames)]
    else:
        print("[ERROR] Especificá --image, --folder o --camera")
        raise SystemExit(1)

    cam_ok, sensor_kind = detect_hardware()
    if args.camera and not cam_ok:
        print("[ERROR] --camera requiere picamera2 (solo Raspberry Pi).")
        raise SystemExit(1)

    # Umbral de no-data: con cámara real no hay bordes negros, así que se apaga.
    nodata_thresh = args.nodata_thresh
    if nodata_thresh is None:
        nodata_thresh = 0 if args.camera else 15

    # ── Modelos ─────────────────────────────────────────────────────────
    seg_path = args.onnx or ("outputs/cansat_seg_terrain_v2.onnx"
                             if Path("outputs/cansat_seg_terrain_v2.onnx").is_file()
                             else "outputs/cansat_seg_deeplabv3plus_mobilenetv2.onnx")
    sess = onnxio.load_required(seg_path, "segmentación",
                                hint="Corré export_v2.py o export_onnx.py (ver MODELS.yaml).")

    # El ONNX de este proyecto tiene entrada FIJA (p. ej. [1,3,320,320]). Antes,
    # pasar --img-size 256 producía un error críptico de ORT/cv2 en el primer
    # frame; ahora se ajusta y se avisa.
    if sess.size_px and sess.size_px != args.img_size:
        print(f"  [i] El ONNX fija la entrada a {sess.size_px}px: "
              f"--img-size {args.img_size} → {sess.size_px}")
        args.img_size = sess.size_px

    # Hilos de OpenCV: en un solo núcleo (Pi Zero v1) el default de OpenCV
    # puede sobre-suscribir y perder tiempo en sincronización.
    if args.cv2_threads > 0:
        cv2.setNumThreads(args.cv2_threads)
    elif (os.cpu_count() or 1) <= 1:
        cv2.setNumThreads(1)

    damage_on = not args.no_damage
    sess_d = sess_d2 = sess_siam = sess_f = sess_fire = sess_sev = None
    if damage_on:
        sess_d = onnxio.load_required(args.damage_onnx, "daño principal",
                                      hint="Corré export_damage_onnx.py.")
        sess_d2 = onnxio.load_required(args.damage2_onnx, "daño two-stage",
                                       hint="Corré export_damage_v3.py.")
        if args.siamese_onnx:
            sess_siam = onnxio.load_optional(args.siamese_onnx, "siamés")
        else:
            print("  [i] Siamés desactivado (pesos ROTO; ver --siamese-onnx).")
        sess_f = onnxio.load_optional(args.flood_onnx, "flood specialist")
        sess_fire = onnxio.load_optional(args.fire_onnx, "fuego/humo")
        sess_sev = onnxio.load_optional(args.severity_onnx, "severidad")

    # Trazabilidad (F6): hashes de los artefactos que realmente corren.
    model_ids = {"seg": _hash8(seg_path)}
    for key, pth in (("damage", args.damage_onnx),
                     ("damage2", args.damage2_onnx),
                     ("flood", args.flood_onnx),
                     ("fire", args.fire_onnx),
                     ("severity", args.severity_onnx)):
        if Path(pth).is_file():
            model_ids[key] = _hash8(pth)

    # ── Detección: backend ──────────────────────────────────────────────
    det_backend = args.det_backend
    if args.no_detect:
        det_backend = "none"
    elif det_backend == "auto":
        det_backend = "yolo"          # comportamiento histórico en PC

    yolo = None
    if det_backend == "yolo":
        vis = "visdrone" in Path(args.det_model).name.lower()
        pids = ({int(v) for v in args.det_person.split(",")} if args.det_person
                else set(VISDRONE_PERSON if vis else COCO_PERSON))
        vids = ({int(v) for v in args.det_vehicles.split(",")} if args.det_vehicles
                else set(VISDRONE_VEHICLES if vis else COCO_VEHICLES))
        yolo = load_yolo(args.det_model, pids, vids)
    elif det_backend == "imx500":
        # El NPU usa índices COCO-91 (person=1), no COCO-80 (person=0).
        from cansat.imx500 import SSD_PERSON, SSD_VEHICLES
        pids = ({int(v) for v in args.det_person.split(",")} if args.det_person
                else set(SSD_PERSON))
        vids = ({int(v) for v in args.det_vehicles.split(",")} if args.det_vehicles
                else set(SSD_VEHICLES))
        if not args.camera:
            print("  [ERROR] --det-backend imx500 requiere --camera (el NPU vive "
                  "en el AI Camera).")
            raise SystemExit(1)
    else:
        pids, vids = set(COCO_PERSON), set(COCO_VEHICLES)

    # Tensor del baseline siamés: se prepara UNA vez, no por frame.
    base_tensor = None
    if sess_siam:
        bp = Path(args.baseline)
        if bp.is_file():
            b0 = cv2.imread(str(bp))
            if b0 is not None:
                base_tensor = PP.preprocess_bgr(b0, args.img_size)
            else:
                print(f"  [WARN] no se pudo leer {bp}; siamés desactivado.")
                sess_siam = None
        else:
            print(f"  [WARN] {bp} no existe; siamés desactivado. "
                  f"Generalo con baseline_tool.py --lat .. --lon ..")
            sess_siam = None

    # ── Banner (una sola vez, no por frame) ─────────────────────────────
    print("=" * 66)
    print("  CanSat La Base — Pipeline de misión v7")
    print(f"  Inferencia   : {'onnxruntime' if _ORT_DISPONIBLE else 'cv2.dnn (sin onnxruntime)'}")
    print(f"  Segmentación : {sess.describe()}")
    print(f"  Daño         : {'OFF (--no-damage)' if not damage_on else 'ON'}")
    if damage_on:
        print(f"                 principal={Path(args.damage_onnx).name} "
              f"two-stage={Path(args.damage2_onnx).name}")
        print(f"                 siamés={'ON' if sess_siam else 'OFF'} "
              f"flood={'ON' if sess_f else 'OFF'} "
              f"fuego/humo={'ON' if sess_fire else 'OFF'} "
              f"severidad={'ON' if sess_sev else 'OFF'}")
    print(f"  Detección    : {'OFF' if det_backend == 'none' else det_backend}"
          + (f" · {args.det_model}" if det_backend == "yolo" else "")
          + (f" · {Path(args.imx500_model).name}" if det_backend == "imx500" else ""))
    print(f"  Daño umbral  : principal/siamés "
          f"{args.damage_threshold if args.damage_threshold is not None else IDX.DAMAGE_CONSENSUS_PCT} %"
          f" · two-stage "
          f"{args.damage_threshold_two_stage if args.damage_threshold_two_stage is not None else IDX.DAMAGE_CONSENSUS_PCT_TWO_STAGE} %"
          + (" (--damage-threshold)" if args.damage_threshold is not None else " (calibrado)"))
    print(f"  Evidencia    : {'OFF' if args.no_vis else Path(args.out_dir) / 'vis'}")
    print(f"  Cámara HW    : {'picamera2' if cam_ok else 'no (usa archivos)'}")
    print(f"  Barómetro    : {sensor_kind or 'no (atmósfera simulada)'}"
          + (" · con humedad" if sensor_kind == "bme280" else ""))
    print("  Enhance/TTA/CRF/Temporal : "
          + " ".join(f"{n}={'ON' if v else 'OFF'}" for n, v in
                     (("enh", args.enhance), ("tta", args.tta),
                      ("crf", args.crf), ("temp", args.temporal))))
    print(f"  No-data      : umbral {nodata_thresh} modo {args.nodata_mode}")
    print(f"  Sharp gate   : {args.sharp_gate or 'OFF'}"
          + (f" (drop={'ON' if args.sharp_drop else 'OFF'})" if args.sharp_gate else ""))
    print(f"  Sampler      : {'OFF' if not args.sampler else f'ON · uncert-max {args.uncert_max}'}")
    print(f"  UART state   : {args.uart_state or 'OFF'}")
    print(f"  Pérdidas est.: densidad {supuestos.pop_density:.0f} hab/km² · "
          f"ocupación {supuestos.occupancy:.2f} · colapso {supuestos.collapse_frac:.2f} · "
          f"letalidad {supuestos.fatality:.2f}")
    if lat_f or lon_f:
        print(f"  GPS          : {lat_f:.5f}, {lon_f:.5f}")
    print("=" * 66)

    # ── Salida ──────────────────────────────────────────────────────────
    out = Path(args.out_dir)
    vis_dir = out / "vis"
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    sampler = AdaptiveSampler(uncert_max=args.uncert_max) if args.sampler else None
    hi_dir, full_dir, th_dir = (out / "high_res", out / "full_res", out / "thumb")
    if sampler is not None:
        for d in (hi_dir, full_dir, th_dir):
            d.mkdir(parents=True, exist_ok=True)
    pri_count: Counter = Counter()

    # Append por defecto: cada corrida abre un archivo nuevo con sufijo horario.
    # Antes se abría con "w" y cada corrida borraba el vuelo anterior.
    # Los handles se cierran en el finally del loop (no se usa `with` porque el
    # ciclo de vida abarca todo el try/except del que depende el cierre).
    if args.overwrite:
        csv_path, jsonl_path = out / "telemetry.csv", out / "telemetry.jsonl"
        mode = "w"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out / f"telemetry_{stamp}.csv"
        jsonl_path = out / f"telemetry_{stamp}.jsonl"
        mode = "w"
    # Symlink/copia canónica para que la estación terrena encuentre siempre el
    # último vuelo en la ruta del contrato.
    latest_csv = out / "telemetry.csv"

    jsonl = open(jsonl_path, mode, encoding="utf-8")
    csvf = open(csv_path, mode, newline="", encoding="utf-8")
    # (handles cerrados en el finally del loop; no se usa `with` porque el
    #  ciclo de vida abarca todo el try/except del que depende el cierre)
    writer = csv.writer(csvf)
    if mode == "w":
        writer.writerow(CSV_COLUMNS)

    # ── Sensores ────────────────────────────────────────────────────────
    bmp = None
    p0 = 1013.25
    if sensor_kind:
        try:
            board = importlib.import_module("board")
            if sensor_kind == "bme280":
                drv = importlib.import_module("adafruit_bme280")
                bmp = drv.Adafruit_BME280_I2C(board.I2C())
            else:
                drv = importlib.import_module("adafruit_bmp280")
                bmp = drv.Adafruit_BMP280_I2C(board.I2C())
            lect = float(bmp.pressure)
            tiene_hum = hasattr(bmp, "humidity")
            if args.p0_alt is not None:
                # Calibración con altitud conocida del predio → QNH local.
                p0 = PROTO.sea_level_pressure(lect, args.p0_alt)
                print(f"  [{sensor_kind.upper()}] {lect:.1f} hPa en {args.p0_alt:.0f} m "
                      f"→ p0={p0:.1f} hPa"
                      + (f" · humedad {float(bmp.humidity):.0f} %" if tiene_hum else ""))
            else:
                p0 = lect
                print(f"  [{sensor_kind.upper()}] p0={p0:.1f} hPa (presión local SIN "
                      f"calibrar; la altitud es relativa al punto de encendido. "
                      f"Usá --p0-alt para altitud absoluta.)"
                      + (f" · humedad {float(bmp.humidity):.0f} %" if tiene_hum else ""))
        except Exception as e:
            print(f"  [WARN] {sensor_kind} no pudo iniciar ({e}). Simulando atmósfera.")
            bmp = None

    cam = None
    if args.camera:
        if det_backend == "imx500":
            # AI Camera: el NPU del IMX500 corre la red de detección y la Pi
            # sólo lee metadata (clave en la Zero v1, sin torch).
            from cansat.imx500 import Imx500Camera

            cam = Imx500Camera(args.imx500_model)
            cam.start()
            print(f"  [IMX500] detección on-sensor: {Path(args.imx500_model).name}")
        else:
            Picamera2 = importlib.import_module("picamera2").Picamera2
            cam = Picamera2()
            cam.configure(cam.create_still_configuration())
            cam.start()
        time.sleep(2)

    # ── Loop ────────────────────────────────────────────────────────────
    t0 = time.time()
    n_pkt = 0
    n_blur = 0
    prev_logits = None
    bmp_fail_streak = 0
    p0_u = None

    try:
        for i, (name, path) in enumerate(frames):
            t = (time.time() - t0) if args.camera else i * args.interval
            t_frame = time.perf_counter()

            boxes = None          # None = hay que detectar; imx500 lo llena al capturar
            if path is not None:
                bgr = cv2.imread(str(path))
            elif det_backend == "imx500":
                bgr, people, veh, boxes = cam.capture()
            else:
                rgb = cam.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if bgr is None:
                print(f"  [WARN] No se pudo leer {name}, salteando.")
                continue

            if args.enhance:
                bgr = enhance_frame(bgr)
            sharp = sharp_score(bgr)
            sharp_ok = (args.sharp_gate <= 0) or (sharp >= args.sharp_gate)
            if not sharp_ok:
                n_blur += 1
                if args.sharp_drop:
                    # --sharp-drop: no se infiere, no va a telemetría ni a evidencia.
                    # Antes la promesa del docstring ("descartar") era falsa: sólo
                    # marcaba y el frame seguía costando 6 modelos por frame.
                    print(f"  [SKIP] {name} borroso (sharp={sharp:.0f} < "
                          f"{args.sharp_gate:.0f}) — descartado")
                    continue

            # ── Segmentación ────────────────────────────────────────────
            # La máscara de válidos se calcula ANTES de inferir: la usan la
            # incertidumbre, los porcentajes de daño y el veredicto.
            valid = valid_mask_for(bgr, args.img_size, nodata_thresh,
                                   args.nodata_mode)
            n_valid = max(1, int(valid.sum()))

            t_seg = time.perf_counter()
            tensor = PP.preprocess_bgr(bgr, args.img_size)
            if args.tta:
                logits, uncert = tta_logits(sess, bgr, args.img_size)
            else:
                logits = sess.run({"input": tensor})
                # Incertidumbre barata siempre disponible (antes: 0.0 sin --tta,
                # lo que dejaba inerte el término de incertidumbre del sampler).
                uncert = entropy_uncertainty(logits, valid)

            if args.crf:
                logits = crf_lite(logits, bgr, iters=args.crf_iters)
            if args.temporal:
                if sharp_ok and prev_logits is not None and prev_logits.shape == logits.shape:
                    logits = (args.temporal_alpha * logits
                              + (1 - args.temporal_alpha) * prev_logits)
                # Sólo se actualiza con frames válidos: antes un borroso se
                # guardaba como estado y contaminaba el suavizado del siguiente.
                if sharp_ok:
                    prev_logits = logits
            ms_seg = (time.perf_counter() - t_seg) * 1000.0

            pcts, dom, seg, valid = terrain_percentages(
                logits, bgr, args.img_size, nodata_thresh, args.nodata_mode,
                valid=valid)
            nodata_pct = 100.0 * (1.0 - valid.mean())

            # ── Detección ───────────────────────────────────────────────
            if boxes is None:
                if yolo is not None:
                    people, veh, boxes = detect_objects(
                        yolo, bgr, args.det_conf, args.det_imgsz, pids, vids)
                else:
                    people, veh, boxes = 0, 0, []

            # ── Atmósfera (con degradación, nunca crashea) ──────────────
            reading, err = (None, None)
            if bmp is not None:
                reading, err = bmp_read(bmp, p0)
                if reading is None:
                    bmp_fail_streak += 1
                    if bmp_fail_streak == 1 or bmp_fail_streak % 10 == 0:
                        print(f"  [WARN] {err} — usando atmósfera simulada "
                              f"(fallas consecutivas: {bmp_fail_streak})")
                else:
                    bmp_fail_streak = 0
            if reading is None and bmp is None and args.uart_state:
                # Sin barómetro en la Pi: usar la última lectura del ESP32 que
                # dejó el listener (incluye humedad del BME280 del DPD).
                us = PROTO.read_uart_state(args.uart_state)
                if us is not None:
                    p_u, temp_u = us["p_hPa"], us["temp_C"]
                    if us["lat"] or us["lon"]:
                        lat_f, lon_f = us["lat"], us["lon"]
                    if p0_u is None:      # calibrar una sola vez con --p0-alt
                        p0_u = (PROTO.sea_level_pressure(p_u, args.p0_alt)
                                if args.p0_alt is not None else 1013.25)
                    reading = (PROTO.pressure_to_altitude(p_u, p0_u), p_u, temp_u,
                               (us.get("hum_pct") or None))
            if reading is None:
                alt, p, temp = PROTO.sim_atmo(t, args.apogee, args.speed)
                hum = None
            else:
                alt, p, temp, hum = reading

            # ── Estrés ambiental (DPD): bruma por imagen + calor por sensores ──
            # El DPD pide estudiar el estrés ambiental "junto con la información
            # de los sensores": acá se combinan la bruma/aerosoles de la imagen
            # (dark channel prior, cansat/stress.py) con el humidex de
            # temperatura y humedad. La bruma se calcula sobre una copia chica
            # (≤320 px) para que en la Pi no cueste.
            haze = None
            if not args.no_stress:
                h, w = bgr.shape[:2]
                esc = STRESS_MAX_SIDE / max(h, w)
                small = (cv2.resize(
                    bgr, (max(1, int(w * esc)), max(1, int(h * esc))),
                    interpolation=cv2.INTER_AREA) if esc < 1.0 else bgr)
                haze = ST.haze_metrics(small)
            hx = ST.humidex(temp, hum) if hum is not None else None

            # Índices y veredicto: FUENTE ÚNICA (cansat.indices).
            env = IDX.environment(pcts, n_green_patches=_green_patches(seg, valid),
                                  valid_frac=float(valid.mean()),
                                  haze_pct=(haze["haze_pct"] if haze else None),
                                  humidex=hx)
            vcode = env["vcode"]

            # ── Daño: consenso real de modelos ──────────────────────────
            # ``None`` = modelo desactivado: NO vota. Antes se pasaba 0.0 y un
            # modelo apagado contaba como un voto en contra del daño.
            #
            # ⚠ Los porcentajes se calculan sobre PÍXELES VÁLIDOS (``n_valid``),
            #   igual que el terreno. Antes se dividían por el frame completo:
            #   con un 50 % de borde negro (tiles) el daño quedaba subestimado
            #   a la mitad y el consenso no podía dispararse nunca.
            pct_dan = pct_dan2 = pct_siam = None
            pct_dan2_edif = None
            pct_flood = pct_fw = None
            pct_fire = pct_smoke = None
            colapso_pct = None
            dan_max = 0.0
            if damage_on:
                t_dmg = time.perf_counter()
                pred_d = np.argmax(sess_d.run({"input": tensor})[0], axis=0)
                pred_d2 = np.argmax(sess_d2.run({"input": tensor})[0], axis=0)
                pct_dan = float(((pred_d == 2) & valid).sum()) / n_valid * 100.0
                bmask = (seg == 1) & valid               # edificios del terreno
                pct_dan2 = float(((pred_d2 == 2) & bmask).sum()) / n_valid * 100.0
                n_edif = int(bmask.sum())
                if n_edif:
                    # Métrica de análisis: daño relativo a los edificios
                    # predichos. Es la magnitud con la que se entrenó el
                    # two-stage (supervisión solo en edificios) y la que hay
                    # que mirar para calibrar, pero NO es la que vota.
                    pct_dan2_edif = float(((pred_d2 == 2) & bmask).sum()) / n_edif * 100.0

                if sess_siam and base_tensor is not None:
                    pred_s = np.argmax(
                        sess_siam.run({"pre": base_tensor, "post": tensor})[0], axis=0)
                    pct_siam = float(((pred_s == 2) & valid).sum()) / n_valid * 100.0

                # El flood specialist se corre UNA sola vez por frame y el
                # resultado se reutiliza para el diagnóstico y para el sampler.
                # Antes se ejecutaba dos veces (rama de diagnóstico + sampler).
                if sess_f:
                    # El flood puede tener su propio tamaño de entrada (p.ej.
                    # 224 px: la mitad de cómputo que 320). Si coincide con el
                    # del terreno se reutiliza el tensor; si no, se preprocesa
                    # aparte y se adapta la máscara de válidos.
                    size_f = sess_f.size_px or args.img_size
                    if size_f == args.img_size:
                        tensor_f, valid_f = tensor, valid
                    else:
                        tensor_f = PP.preprocess_bgr(bgr, size_f)
                        valid_f = cv2.resize(
                            valid.astype(np.uint8), (size_f, size_f),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                    pred_f = np.argmax(sess_f.run({"input": tensor_f})[0], axis=0)
                    n_valid_f = max(1, int(valid_f.sum()))
                    pct_flood = float(((pred_f == 1) & valid_f).sum()) / n_valid_f * 100.0
                    pct_fw = float(((pred_f == 2) & valid_f).sum()) / n_valid_f * 100.0

                # Fuego/humo (F3): alerta independiente del daño estructural.
                if sess_fire:
                    size_fire = sess_fire.size_px or args.img_size
                    if size_fire == args.img_size:
                        tensor_fire, valid_fire = tensor, valid
                    else:
                        tensor_fire = PP.preprocess_bgr(bgr, size_fire)
                        valid_fire = cv2.resize(
                            valid.astype(np.uint8), (size_fire, size_fire),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                    pred_fire = np.argmax(
                        sess_fire.run({"input": tensor_fire})[0], axis=0)
                    n_valid_fire = max(1, int(valid_fire.sum()))
                    pct_fire = float(((pred_fire == 1) & valid_fire).sum()) / n_valid_fire * 100.0
                    pct_smoke = float(((pred_fire == 2) & valid_fire).sum()) / n_valid_fire * 100.0

                # Severidad (F2b): fracción de colapso MEDIDA para casualties.
                # colapso = (mayor + destruido) / (cualquier edificio) del
                # modelo de severidad. Reemplaza el supuesto 0.3 cuando existe.
                if sess_sev:
                    size_sev = sess_sev.size_px or args.img_size
                    tensor_sev = (tensor if size_sev == args.img_size
                                  else PP.preprocess_bgr(bgr, size_sev))
                    pred_sev = np.argmax(
                        sess_sev.run({"input": tensor_sev})[0], axis=0)
                    n_edif_sev = int((pred_sev >= 1).sum())
                    if n_edif_sev:
                        colapso_pct = float((pred_sev >= 3).sum()) / n_edif_sev * 100.0

                dan_max = max(v for v in (pct_dan, pct_dan2, pct_siam)
                              if v is not None)
                ms_dmg = (time.perf_counter() - t_dmg) * 1000.0
            else:
                ms_dmg = 0.0

            diag, alert = IDX.diagnose(
                pcts, pct_dan, pct_dan2, pct_siam,
                pct_flood=pct_flood, pct_flood_water=pct_fw,
                flood_available=bool(sess_f),
                pct_fire=pct_fire, pct_smoke=pct_smoke,
                consensus_pct=args.damage_threshold,
                consensus_pct_two_stage=args.damage_threshold_two_stage,
            )

            # Área afectada en m² según la huella en tierra del frame, y
            # estimación de exposición/pérdidas humanas (DPD: "estimar las
            # posibles pérdidas humanas resultantes").
            area_frame_m2 = ground_area_m2(alt)
            aff_m2 = area_frame_m2 * dan_max / 100.0
            # Densidad poblacional REAL por GPS (WorldPop, grilla 0.1°) cuando
            # hay fix; si no, el supuesto de --pop-density (la telemetría
            # declara cuál se usó).
            dens_pob, pob_fuente = POP.densidad(lat_f, lon_f,
                                                default=args.pop_density)
            sup_frame = (supuestos if pob_fuente == "supuesto"
                         else Supuestos(pop_density=dens_pob,
                                        occupancy=supuestos.occupancy,
                                        collapse_frac=supuestos.collapse_frac,
                                        fatality=supuestos.fatality))
            est = estimar_perdidas(
                dan_max, area_frame_m2, sup_frame,
                collapse_frac_medido=(colapso_pct / 100.0
                                      if colapso_pct is not None else None))

            # ── Muestreo adaptativo ─────────────────────────────────────
            if sampler is not None:
                score = sampler.interest_score(
                    pcts, env["usi"], uncert=uncert,
                    pct_dan=dan_max, pct_flood=(pct_flood or 0.0))
                score = min(1.0, score + sampler.coverage_boost(pcts))
                dec = sampler.decide(score, uncert=uncert)
                pri_count[dec["priority"]] += 1
                # Un frame borroso nunca merece alta prioridad — y tampoco
                # "revisión humana": con review=True, decide() habría devuelto
                # HIGH y el estado quedaba contradictorio (LOW + review).
                if not sharp_ok and dec["action"] != "thumb":
                    dec = {"priority": "LOW", "action": "thumb",
                           "review": False, "score": dec["score"]}
                dst = {"high_res": hi_dir, "full_res": full_dir,
                       "thumb": th_dir}[dec["action"]]
                if dec["action"] == "thumb":
                    cv2.imwrite(str(dst / f"{name}.jpg"),
                                cv2.resize(bgr, (256, 256)),
                                [cv2.IMWRITE_JPEG_QUALITY, 70])
                else:
                    cv2.imwrite(str(dst / f"{name}.png"), bgr)
            else:
                dec = {"priority": "FULL", "score": 0.0, "review": False}

            # ── Evidencia visual ────────────────────────────────────────
            if not args.no_vis:
                vis = build_evidence(
                    bgr, seg, boxes, env, people, veh, alt, p, temp, t,
                    nodata_pct, nodata_thresh, person_ids=frozenset(pids),
                    diag=diag, sharp=sharp, sharp_ok=sharp_ok)
                cv2.imwrite(str(vis_dir / f"{name}_evid.jpg"), vis)

            # ── Registro ────────────────────────────────────────────────
            pkt = {
                "t_s": round(t, 1), "alt_m": round(alt, 1),
                "p_hPa": round(p, 1), "temp_c": round(temp, 2),
                "hum_pct": (round(hum, 1) if hum is not None else None),
                "terrain": {n: round(v, 1)
                            for n, v in zip(CLASS_NAMES, pcts, strict=False)},
                "dominant": CLASS_NAMES[dom] if 0 <= dom < len(CLASS_NAMES) else None,
                "usi": round(env["usi"], 2),
                "usi_norm": round(env["usi_norm"], 4),
                "gvi": round(env["gvi"], 3),
                "ndvi": round(env["gvi"], 3),   # clave legacy del contrato
                "verdict": env["verdict"],
                "people": people, "vehicles": veh,
                "danado_pct": _rnum(pct_dan),
                "danado2_pct": _rnum(pct_dan2),
                "danado2_edif_pct": _rnum(pct_dan2_edif),
                "danado_siam_pct": _rnum(pct_siam),
                "danado_max_pct": round(dan_max, 1),
                "fire_pct": _rnum(pct_fire),
                "smoke_pct": _rnum(pct_smoke),
                "haze_pct": (haze["haze_pct"] if haze else None),
                "visibility": (haze["visibility"] if haze else None),
                "humidex": hx,
                "contam": env.get("contam"),
                "heat": env.get("heat"),
                "stress_idx": env.get("stress_idx"),
                # Trazabilidad F6: modelo exacto (hash) y precisión por frame.
                "model_ids": model_ids, "quant": "fp32",
                "aff_m2": int(aff_m2),
                "area_m2": int(area_frame_m2),
                # Estimación de pérdidas humanas (ver cansat/casualties.py):
                # exposición con supuestos declarados, no predicción.
                "personas_afectadas": est.personas_afectadas,
                "perdidas_est": est.perdidas_estimadas,
                "perdidas_min": est.perdidas_min,
                "perdidas_max": est.perdidas_max,
                "colapso_pct": _rnum(colapso_pct),
                "colapso_fuente": est.colapso_fuente,
                # Densidad poblacional del frame y de dónde salió (DPD: pérdidas).
                "pop_density": round(dens_pob, 1),
                "pop_fuente": pob_fuente,
                "lat": round(lat_f, 5), "lon": round(lon_f, 5),
                "diag": diag, "alert": alert,
                "sharp": round(sharp, 1), "sharp_ok": bool(sharp_ok),
                "uncert": round(uncert, 5),
                "nodata_pct": round(nodata_pct, 1),
                # Tiempos por etapa: en la Pi Zero v1 son el dato que define
                # --frames/--interval (antes no se medían en ningún lado).
                "ms_seg": round(ms_seg, 1), "ms_dmg": round(ms_dmg, 1),
                "ms_total": round((time.perf_counter() - t_frame) * 1000.0, 1),
                "sample_pri": dec["priority"], "sample_score": dec["score"],
                "src": name,
            }
            jsonl.write(json.dumps(pkt, ensure_ascii=False) + "\n")
            # La columna danado_pct del CSV lleva el MÁXIMO del consenso
            # (dan_max), no sólo el modelo principal: aff_m2, diag y alert ya se
            # calculan con el máximo, y antes la estación terrena mostraba un
            # daño distinto del que el vuelo usó para decidir. El detalle por
            # modelo queda en el JSONL.
            writer.writerow([
                pkt["t_s"], pkt["alt_m"], pkt["p_hPa"], pkt["temp_c"],
                *[round(v, 1) for v in pcts],
                dom, round(env["usi"], 2), round(env["gvi"], 3),
                env["verdict"], people, veh,
                round(dan_max, 1), int(aff_m2), diag, alert,
                round(sharp, 1), name, dec["priority"], round(dec["score"], 3),
                # Extensiones DPD (mismo orden que CSV_COLUMNS)
                round(lat_f, 5), round(lon_f, 5),
                (round(hum, 1) if hum is not None else ""),
                int(area_frame_m2),
                est.personas_afectadas, est.perdidas_estimadas,
                (round(pct_fire, 1) if pct_fire is not None else ""),
                (round(pct_smoke, 1) if pct_smoke is not None else ""),
                (haze["haze_pct"] if haze else ""),
                (hx if hx is not None else ""),
                (env.get("stress_idx") if env.get("stress_idx") is not None else ""),
                (round(colapso_pct, 1) if colapso_pct is not None else ""),
            ])
            csvf.flush()

            # ── Radio ───────────────────────────────────────────────────
            if args.radio_format != "off":
                rp = PROTO.Packet(
                    pkt=n_pkt, t_s=t, alt_m=alt, p_hPa=p, temp_C=temp,
                    veg=pcts[0], bui=pcts[1], wat=pcts[2], bare=pcts[3],
                    oth=pcts[4], dom=dom, usi=env["usi"], gvi=env["gvi"],
                    vcode=vcode, personas=people, vehiculos=veh, alert=alert,
                    danado_pct=dan_max, lat=lat_f, lon=lon_f,
                    hum_pct=(hum or 0.0))
                radio = (PROTO.format_packet(rp) if args.radio_format == "v2"
                         else PROTO.format_legacy_v1(rp))
            else:
                radio = "(radio off)"

            print(f"  {radio}")
            print(f"           → {env['verdict']} (USI {env['usi']:.2f}, "
                  f"GVI {env['gvi']:+.3f}) | {people} personas, {veh} vehículos")
            print(f"           → {diag} | daño {_disp(pct_dan)}% / 2stage "
                  f"{_disp(pct_dan2)}% / siamés {_disp(pct_siam)}% "
                  f"(~{int(aff_m2)} m²) ALERTA={alert}"
                  + (f" | afectados≈{est.personas_afectadas:.1f} "
                     f"pérdidas≈{est.perdidas_estimadas:.2f} "
                     f"[{est.perdidas_min:.2f}-{est.perdidas_max:.2f}]"
                     if dan_max > 0 else "")
                  + ("" if sharp_ok else f" | BORROSO sharp={sharp:.0f}"))
            n_pkt += 1

            if args.camera and i < len(frames) - 1:
                time.sleep(args.interval)
    finally:
        jsonl.close()
        csvf.close()
        if cam:
            with contextlib.suppress(Exception):
                cam.stop()

    # La estación terrena lee siempre outputs/mission/telemetry.csv: dejamos ahí
    # una copia del vuelo recién terminado (sin pisar el histórico versionado).
    # También el JSONL: la estación lo usa para el panel de muestreo (uncert,
    # tiempos por etapa, nodata por frame), que no están en el CSV.
    if not args.overwrite and csv_path != latest_csv:
        try:
            latest_csv.write_bytes(csv_path.read_bytes())
        except OSError as e:
            print(f"  [WARN] no se pudo actualizar {latest_csv}: {e}")
        latest_jsonl = out / "telemetry.jsonl"
        try:
            latest_jsonl.write_bytes(jsonl_path.read_bytes())
        except OSError as e:
            print(f"  [WARN] no se pudo actualizar {latest_jsonl}: {e}")

    print("=" * 66)
    print(f"  {n_pkt} paquetes de telemetría generados")
    if n_blur:
        if args.sharp_drop:
            print(f"  {n_blur} frame(s) descartados por borrosos "
                  f"(sharp < {args.sharp_gate:.0f}) — no entraron a telemetría")
        else:
            print(f"  {n_blur} frame(s) bajo el umbral de nitidez "
                  f"({args.sharp_gate:.0f}) — marcados, no descartados "
                  f"(usá --sharp-drop para descartarlos)")
    if sampler is not None:
        print(f"  Sampler: {dict(pri_count)} "
              f"(HIGH→high_res / MEDIUM→full_res / LOW→thumb)")
    if not args.no_vis:
        print(f"  Evidencia visual: {vis_dir.resolve()}")
    print(f"  Logs: {csv_path}")
    print(f"        {jsonl_path}")
    if not args.overwrite:
        print(f"  Contrato (último vuelo): {latest_csv}")
    print("=" * 66)
    return 0


def _green_patches(seg, valid, min_px: int = 100) -> int:
    """Parches de vegetación >= min_px (para la fragmentación del verde)."""
    veg = ((seg == 0) & valid).astype(np.uint8)
    if not veg.any():
        return 0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(veg, connectivity=8)
    if n <= 1:
        return 0
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_px).sum())


if __name__ == "__main__":
    raise SystemExit(main())
