#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_server.py — Estación Terrena Web · Misión CanSat LB135
===========================================================

Servidor único (solo stdlib de Python) que cumple dos roles:

    1. Servir los estáticos de la carpeta ``web/``  (index.html, style.css, app.js).
  2. Exponer una API JSON de **solo lectura** sobre los artefactos del proyecto:

        GET /api/mission      -> telemetry.csv parseado + summary.json + mapa de archivos
        GET /api/frame/<src>  -> una fila del CSV + rutas de imágenes disponibles
        GET /api/consulta     -> Consulta Terrestre (q=…&poly=lon,lat;…) vía el motor simbólico
        GET /api/gradcam      -> explicabilidad del daño (src=…&modelo=…&clase=…), cacheada
        GET /api/health       -> estado del servidor (para el indicador LIVE de la web)
        GET /img/<relpath>    -> proxy de imágenes desde la raíz del proyecto
                                 (mimetype correcto + sin caché, para live refresh)

No hay base de datos: todo se lee del disco en cada petición (con una caché
ligera invalidada por mtime para no escanear el árbol 20 veces por segundo).

Uso:
    python web_server.py                 # http://localhost:8000
    python web_server.py --port 8080     # otro puerto
    python web_server.py --root /ruta    # otra raíz de proyecto

Requisitos: Python 3.8+ (sin dependencias externas).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# --------------------------------------------------------------------------- #
# Configuración de rutas del proyecto (contrato de datos, solo lectura)
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent          # raíz del proyecto
WEB_DIR = ROOT / "web"                          # estáticos legacy (fallback)
DIST_DIR = ROOT / "web-app" / "dist"            # build de Vite (preferido si existe)

TELEMETRY_CSV = ROOT / "outputs" / "mission" / "telemetry.csv"
CORRIDOR_MAP = ROOT / "outputs" / "corridor_map.jpg"
SUMMARY_JSON = ROOT / "entrega" / "summary.json"

MISSION_DIR = ROOT / "outputs" / "mission"
ENTREGA_DIR = ROOT / "entrega"
# Grad-CAM: PNGs de explicabilidad generados por el repo de vuelo (cache local).
GRADCAM_DIR = ROOT / "outputs" / "gradcam"

# Consulta Terrestre: el motor simbólico vive en el repo de vuelo (cansat/consultas.py)
# y se invoca por subproceso para que esta estación siga siendo stdlib-only.
FLIGHT_ROOT = ROOT.parent / "cansat_seg_poc"
CONSULTA_PYTHON: str | None = None      # None = venv del repo de vuelo o sys.executable

# Columnas esperadas del CSV de telemetría (contrato). Si falta alguna, se
# rellena con None y la web degrada elegantemente en lugar de romperse.
CSV_COLUMNS = [
    "t_s", "alt_m", "p_hpa", "temp_c",
    "veg", "bui", "wat", "bare", "oth", "dom",
    "usi", "ndvi", "verdict",
    "people", "vehicles", "danado_pct", "aff_m2", "diag", "alert",
    "sharp", "src", "sample_pri", "sample_score",
    # Extensiones DPD (GPS + humedad + estimación de pérdidas humanas)
    "lat", "lon", "hum_pct", "area_m2", "personas_afectadas", "perdidas_est",
    # F3 (2026-09-18): fuego/humo (vacío si el modelo no corrió)
    "fire_pct", "smoke_pct",
    # Estrés ambiental (2026-09-18): bruma por imagen + calor por sensores
    "haze_pct", "humidex", "stress_idx",
    # Severidad (2026-09-18): fracción de colapso medida (vacío = supuesto)
    "colapso_pct",
]

# Campos numéricos -> float. El resto se deja como string.
FLOAT_FIELDS = {
    "t_s", "alt_m", "p_hpa", "temp_c",
    "veg", "bui", "wat", "bare", "oth",
    "usi", "ndvi", "danado_pct", "aff_m2", "sharp", "sample_score",
    "lat", "lon", "hum_pct", "personas_afectadas", "perdidas_est",
    "fire_pct", "smoke_pct", "haze_pct", "humidex", "stress_idx",
    "colapso_pct",
}
INT_FIELDS = {"people", "vehicles", "alert", "area_m2"}

# Extensiones de imagen aceptadas, en orden de preferencia, para cada bucket.
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def _configure(root: Path) -> None:
    """
    (Re)asigna TODAS las rutas del contrato a partir de una raiz.

    Unica fuente de verdad: se llama al importar (con la raiz del script) y con
    ``--root``. Antes el bloque de ``--root`` duplicaba todo a mano.
    """
    global ROOT, WEB_DIR, DIST_DIR, TELEMETRY_CSV, CORRIDOR_MAP, SUMMARY_JSON
    global MISSION_DIR, ENTREGA_DIR, GRADCAM_DIR, BUCKETS, TELEMETRY_JSONL, FLIGHT_ROOT

    # ¿El FLIGHT_ROOT era el default relativo a la ROOT vieja? Se mide ANTES de
    # reasignar ROOT; si no, la comparación se hace contra la ROOT nueva y el
    # default nunca se recalcula para --root.
    era_default = FLIGHT_ROOT == ROOT.parent / "cansat_seg_poc"

    ROOT = root
    WEB_DIR = ROOT / "web"
    DIST_DIR = ROOT / "web-app" / "dist"
    MISSION_DIR = ROOT / "outputs" / "mission"
    ENTREGA_DIR = ROOT / "entrega"
    GRADCAM_DIR = ROOT / "outputs" / "gradcam"
    TELEMETRY_CSV = MISSION_DIR / "telemetry.csv"
    TELEMETRY_JSONL = MISSION_DIR / "telemetry.jsonl"
    CORRIDOR_MAP = ROOT / "outputs" / "corridor_map.jpg"
    SUMMARY_JSON = ENTREGA_DIR / "summary.json"
    if era_default:
        FLIGHT_ROOT = root.parent / "cansat_seg_poc"
    BUCKETS = (
        ("vis",            MISSION_DIR / "vis",       "_evid"),
        ("high_res",       MISSION_DIR / "high_res",  ""),
        ("full_res",       MISSION_DIR / "full_res",  ""),
        ("thumb",          MISSION_DIR / "thumb",     ""),
        ("ens_seg",        ENTREGA_DIR / "ens_seg",   "_b5"),
        ("enhanced",       ENTREGA_DIR / "enhanced",  "_edsr"),
        # Máscaras de clase por frame (Consulta Terrestre; ver entrega/masks).
        ("masks_terreno",  ENTREGA_DIR / "masks",     "_terreno"),
        ("masks_dano2",    ENTREGA_DIR / "masks",     "_dano2"),
        ("masks_flood",    ENTREGA_DIR / "masks",     "_flood"),
        ("masks_vias",     ENTREGA_DIR / "masks",     "_vias"),
    )


# Buckets de imágenes del pipeline. Cada entrada: (clave API, carpeta, sufijo).
#   vis        -> outputs/mission/vis/<src>_evid.jpg      (frame anotado)
#   high_res   -> outputs/mission/high_res/<src>.*
#   full_res   -> outputs/mission/full_res/<src>.*
#   thumb      -> outputs/mission/thumb/<src>.*
#   ens_seg    -> entrega/ens_seg/<src>_b5.png            (post-vuelo, opcional)
#   enhanced   -> entrega/enhanced/<src>_edsr.jpg         (post-vuelo, opcional)
#   masks_*    -> entrega/masks/<src>_<fuente>.png        (consulta, opcional)
BUCKETS = (
    ("vis",            MISSION_DIR / "vis",       "_evid"),
    ("high_res",       MISSION_DIR / "high_res",  ""),
    ("full_res",       MISSION_DIR / "full_res",  ""),
    ("thumb",          MISSION_DIR / "thumb",     ""),
    ("ens_seg",        ENTREGA_DIR / "ens_seg",   "_b5"),
    ("enhanced",       ENTREGA_DIR / "enhanced",  "_edsr"),
    ("masks_terreno",  ENTREGA_DIR / "masks",     "_terreno"),
    ("masks_dano2",    ENTREGA_DIR / "masks",     "_dano2"),
    ("masks_flood",    ENTREGA_DIR / "masks",     "_flood"),
    ("masks_vias",     ENTREGA_DIR / "masks",     "_vias"),
)

_configure(ROOT)      # única fuente de verdad (ver la función más arriba)

# Aseguramos mimetypes correctos aunque el sistema no los traiga registrados.
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")

# Icono del sitio en SVG inline: evita un 404 ruidoso en la consola del navegador.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0b0e14"/>'
    '<circle cx="16" cy="16" r="8" fill="none" stroke="#3ddc84" stroke-width="2"/>'
    '<circle cx="16" cy="16" r="2.6" fill="#ffb020"/>'
    '<path d="M16 2v5M16 25v5M2 16h5M25 16h5" stroke="#3ddc84" stroke-width="2"/>'
    "</svg>"
).encode("utf-8")


# --------------------------------------------------------------------------- #
# Utilidades de lectura de artefactos
# --------------------------------------------------------------------------- #

def _rel(p: Path) -> str:
    """Ruta relativa a ROOT en formato POSIX (la web la monta como /img/<rel>)."""
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _to_number(field: str, raw: str):
    """Convierte un valor del CSV a float/int, tolerante a basura y vacíos."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        if field in INT_FIELDS:
            # "1.0" -> 1 ; "0" -> 0
            return int(float(s))
        return float(s)
    except (TypeError, ValueError):
        return None


def read_telemetry(path: Path | None = None) -> list[dict]:
    """
    Lee telemetry.csv y devuelve una lista de dicts normalizados.

    * Codificación tolerante a BOM (utf-8-sig) y separador detectado.
    * Columnas ausentes -> None.
    * Columnas extra del CSV -> se conservan (la web las ignora).
    * Se fuerza la presencia de ``src`` como identificador de frame.

    Nota: el path se resuelve DENTRO de la función (y no como default del
    argumento) para respetar ``--root`` en tiempo de ejecución.
    """
    path = Path(path) if path is not None else TELEMETRY_CSV
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel  # type: ignore[assignment]
        reader = csv.DictReader(fh, dialect=dialect)
        raw_rows = list(reader)

    frames: list[dict] = []
    for i, row in enumerate(raw_rows):
        # Normalizamos claves: sin espacios y en minúsculas.
        clean = {}
        for k, v in row.items():
            if k is None:
                continue
            clean[str(k).strip().lower()] = v.strip() if isinstance(v, str) else v

        rec: dict = {}
        for col in CSV_COLUMNS:
            val = clean.get(col)
            rec[col] = _to_number(col, val) if col in FLOAT_FIELDS | INT_FIELDS else (
                None if val in (None, "") else str(val)
            )
        # Campos adicionales no contemplados en el contrato.
        for k, v in clean.items():
            if k not in rec:
                rec[k] = v

        # Identificador: si el CSV no trae src, se sintetiza por índice.
        if not rec.get("src"):
            rec["src"] = f"frame_{i:04d}"
        rec["_idx"] = i          # orden físico en el CSV (posición en el corredor)
        frames.append(rec)

    return frames


def find_image(directory: Path, src: str, suffix: str = "") -> Path | None:
    """Busca ``<directory>/<src><suffix><ext>`` con cualquier extensión de imagen."""
    if not directory.is_dir() or not src:
        return None
    for ext in IMG_EXTS:
        cand = directory / f"{src}{suffix}{ext}"
        if cand.is_file():
            return cand
    return None


def files_for_frame(src: str) -> dict[str, str | None]:
    """Mapa clave -> ruta relativa (o None si el archivo no existe)."""
    out: dict[str, str | None] = {}
    for key, directory, suffix in BUCKETS:
        found = find_image(directory, src, suffix)
        out[key] = _rel(found) if found else None
    return out


def normalize_summary(raw: dict | None) -> dict | None:
    """
    Lleva un ``entrega/summary.json`` de cualquier versión al schema v2 del
    contrato (espejo de ``cansat/summary.py::normalize_summary`` del proyecto
    de vuelo, sin importarlo: son proyectos separados).

    ARREGLADO: antes ``read_summary()`` devolvía el JSON crudo. El productor
    legacy escribía ``alertas: list[str]`` y ``archivos`` plano (rutas y
    conteos mezclados); el frontend tipaba ``alertas: unknown[]`` y solo usaba
    ``.length``. Acá se normaliza: alertas siempre ``list[dict]`` con ``src``,
    archivos siempre ``{rutas, conteos}``.
    """
    if not isinstance(raw, dict):
        return None

    out = dict(raw)

    # alertas: list[str] | list[dict] | cualquier cosa → list[dict] con src.
    alertas = out.get("alertas")
    if isinstance(alertas, list):
        norm = []
        for a in alertas:
            if isinstance(a, dict) and a.get("src"):
                norm.append(a)
            else:
                norm.append({"src": str(a)})
        out["alertas"] = norm

    # archivos: plano → {rutas, conteos}.
    arch = out.get("archivos")
    if isinstance(arch, dict) and "rutas" not in arch and "conteos" not in arch:
        alias = {
            "corredor": "corridor_map",
            "evidencias": "vis",
            "seg_b5": "ens_seg",
            "telemetria_csv": "telemetry",
        }
        rutas: dict[str, str | None] = {}
        conteos: dict[str, int | None] = {}
        for k, v in arch.items():
            key = alias.get(k, k)
            if isinstance(v, bool):
                conteos[key] = int(v)
            elif isinstance(v, int):
                conteos[key] = v
            elif isinstance(v, (list, tuple)):
                conteos[key] = len(v)
                rutas[key] = v[0] if v else None
            elif v is None:
                rutas[key] = None
            else:
                rutas[key] = str(v)
        out["archivos"] = {"rutas": rutas, "conteos": conteos}

    # El schema canónico actual es 3 (ver cansat/summary.py del proyecto de
    # vuelo y summarySchema.ts): v3 agrega el bucket masks de la consulta.
    out.setdefault("schema_version", 3)
    out.setdefault("mision", "LB135")
    out.setdefault("veredictos", {})
    return out


def read_summary() -> dict | None:
    """Lee y normaliza entrega/summary.json. None si no existe o está corrupto."""
    if not SUMMARY_JSON.is_file():
        return None
    try:
        with SUMMARY_JSON.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return normalize_summary(data) if isinstance(data, dict) else {"raw": data}
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Caché ligera del payload /api/mission (invalidada por mtime)
# --------------------------------------------------------------------------- #

class MissionCache:
    """
    Evita re-parsear el CSV y re-escanear los directorios de imágenes en cada
    polling de 3 s. La firma se compone del mtime/tamaño del CSV, de summary.json,
    del mapa de corredor y de los directorios de imágenes (cambian cuando se
    agrega/borra un archivo).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sig: str = ""
        self._payload: dict | None = None

    @staticmethod
    def _stat_sig(p: Path) -> str:
        try:
            st = p.stat()
            return f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            return "0:0"

    @classmethod
    def _dir_sig(cls, directory: Path, limit: int = 512) -> str:
        """
        Firma de un directorio de imagenes.

        ARREGLADO: antes se firmaba con ``_stat_sig(directory)``, o sea el mtime
        DEL DIRECTORIO, que solo cambia al CREAR o BORRAR archivos. El pipeline
        en vivo reescribe los frames con los mismos nombres, asi que la firma no
        cambiaba, la cache no se invalidaba y la web mostraba datos viejos: justo
        el caso "live" que el SSE pretende resolver.

        Ahora se firma con mtime+tamano de cada archivo, acotado a ``limit``
        entradas para no hacer stat de miles de archivos en cada poll de 3 s.
        """
        if not directory.is_dir():
            return "-"
        try:
            entries = sorted(directory.iterdir(), key=lambda q: q.name)
        except OSError:
            return "-"
        if len(entries) > limit:
            return f"dir:{cls._stat_sig(directory)}:{len(entries)}"
        return ";".join(cls._stat_sig(e) for e in entries if e.is_file())

    def _signature(self) -> str:
        parts = [
            self._stat_sig(TELEMETRY_CSV),
            self._stat_sig(SUMMARY_JSON),
            self._stat_sig(CORRIDOR_MAP),
        ]
        for _key, directory, _suffix in BUCKETS:
            parts.append(self._dir_sig(directory))
        return "|".join(parts)

    def get(self) -> dict:
        sig = self._signature()
        with self._lock:
            if self._payload is not None and sig == self._sig:
                # Copia superficial: el payload se trata como solo lectura.
                payload = dict(self._payload)
                payload["cached"] = True
                return payload
            payload = build_mission_payload()
            self._payload = payload
            self._sig = sig
            out = dict(payload)
            out["cached"] = False
            return out


CACHE = MissionCache()


# --------------------------------------------------------------------------- #
# JSONL de telemetría: campos por frame que no están en el CSV
# --------------------------------------------------------------------------- #

# Claves del JSONL que la estación usa para el panel de Muestreo. El CSV ya
# trae sample_pri/sample_score; el JSONL agrega la incertidumbre, los tiempos
# por etapa y el % de píxeles sin datos que el pipeline registra a bordo.
_JSONL_KEYS = ("uncert", "ms_seg", "ms_dmg", "ms_total", "nodata_pct",
               "danado2_edif_pct", "sharp_ok", "area_m2", "supuestos",
               # Contrato v3: trazabilidad de fuentes y modelos por frame.
               "colapso_fuente", "pop_fuente", "ocupacion_fuente",
               "tipo_desastre", "tipo_conf", "model_ids", "quant",
               # Política fire_only_v1 del clasificador de tipo (2026-09-20).
               "tipo_estado", "tipo_politica", "tipo_top1_crudo",
               "tipo_top1_conf_cruda", "tipo_umbral_incendio",
               "tipo_es_confiable", "tipo_modelo_hash", "tipo_notas",
               # OOD/drift (2026-09-21): aviso de fuera-de-distribución por frame.
               "ood_score", "ood_flag")

_JSONL_CACHE: dict = {"sig": "", "extras": {}, "resumen": {}}


def _stat_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "0:0"


def read_jsonl_extras() -> dict:
    """
    Lee ``outputs/mission/telemetry.jsonl`` (el que deja el pipeline) y arma:

      · ``extras``: por ``src``, los campos que no están en el CSV;
      · ``resumen``: agregados para el panel de Muestreo (tiempos, incertidumbre,
        frames borrosos, distribución de prioridad y cobertura por clase).

    Cacheado por mtime+tamaño, igual que el payload principal.
    """
    sig = _stat_sig(TELEMETRY_JSONL)
    if _JSONL_CACHE["sig"] == sig and sig != "0:0":
        return _JSONL_CACHE

    extras: dict[str, dict] = {}
    ms_total: list[float] = []
    uncert: list[float] = []
    nodata: list[float] = []
    pri: dict[str, int] = {}
    n_blur = n_frames = 0
    if TELEMETRY_JSONL.is_file():
        try:
            with TELEMETRY_JSONL.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = str(d.get("src") or "")
                    if not src:
                        continue
                    n_frames += 1
                    extras[src] = {k: d.get(k) for k in _JSONL_KEYS}
                    for lst, key in ((ms_total, "ms_total"), (uncert, "uncert"),
                                     (nodata, "nodata_pct")):
                        v = d.get(key)
                        if isinstance(v, (int, float)):
                            lst.append(float(v))
                    p = d.get("sample_pri")
                    if isinstance(p, str):
                        pri[p] = pri.get(p, 0) + 1
                    if d.get("sharp_ok") is False:
                        n_blur += 1
        except OSError:
            pass

    def _pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return round(s[idx], 3)

    resumen = {
        "n_frames": n_frames,
        "ms_total_mediana": _pct(ms_total, 0.5),
        "ms_total_p95": _pct(ms_total, 0.95),
        "uncert_p50": _pct(uncert, 0.5),
        "uncert_p95": _pct(uncert, 0.95),
        "nodata_p95": _pct(nodata, 0.95),
        "prioridad": pri,
        "n_borrosos": n_blur,
    }
    _JSONL_CACHE.update({"sig": sig, "extras": extras, "resumen": resumen})
    return _JSONL_CACHE


def build_mission_payload() -> dict:
    """Arma el JSON completo de /api/mission."""
    t0 = time.perf_counter()
    frames = read_telemetry()

    # Existencia de imágenes por frame (una sola pasada).
    for fr in frames:
        fr["files"] = files_for_frame(fr.get("src") or "")

    summary = read_summary()

    # Recuento de buckets disponibles, para que la web sepa qué secciones mostrar.
    buckets_available = {key: False for key, _d, _s in BUCKETS}
    for fr in frames:
        for key, rel in (fr.get("files") or {}).items():
            if rel:
                buckets_available[key] = True

    payload = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mision": (summary or {}).get("mision") or "LB135",
        "n_frames": len(frames),
        "frames": frames,
        "columns": CSV_COLUMNS,
        "summary": summary,                     # None si aún no hay post-vuelo
        "assets": {
            "corridor_map": _rel(CORRIDOR_MAP) if CORRIDOR_MAP.is_file() else None,
        },
        "buckets": buckets_available,
        "paths": {
            "telemetry": _rel(TELEMETRY_CSV) if TELEMETRY_CSV.is_file() else None,
            "summary": _rel(SUMMARY_JSON) if SUMMARY_JSON.is_file() else None,
        },
        "build_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    return payload


def _consulta_script() -> Path:
    return FLIGHT_ROOT / "tools" / "consulta.py"


def _consulta_python() -> str:
    """Intérprete del motor: el venv del repo de vuelo si existe, si no el actual."""
    if CONSULTA_PYTHON:
        return CONSULTA_PYTHON
    for cand in (FLIGHT_ROOT / "venv" / "Scripts" / "python.exe",
                 FLIGHT_ROOT / "venv" / "bin" / "python"):
        if cand.is_file():
            return str(cand)
    return sys.executable


def _mtime_ns(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def run_consulta(q: str, region: list | None = None) -> dict:
    """
    Ejecuta una consulta espacial con el motor simbólico del repo de vuelo.

    Subproceso para mantener esta estación sin numpy/cv2. La consulta viaja como
    argumento (sin shell), así que no hay inyección. Respuesta siempre JSON.
    """
    script = _consulta_script()
    if not script.is_file():
        return {"ok": False, "consulta_espacial": False,
                "error": f"no encontré {script}; pasá --flight-root al repo de vuelo"}
    if not q.strip():
        return {"ok": True, "consulta_espacial": False, "soportada": False,
                "consulta": q, "motivo": "consulta vacía"}

    cmd = [_consulta_python(), str(script), "--q", q, "--json",
           "--masks", str(ENTREGA_DIR / "masks"),
           "--telemetry", str(TELEMETRY_CSV),
           # Detecciones con posición (Consulta v2). Si no existe, el motor
           # cae al conteo por frame de la telemetría y lo declara.
           "--jsonl", str(TELEMETRY_JSONL)]
    if region:
        cmd += ["--region", json.dumps(region, ensure_ascii=False)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            timeout=60)
    except subprocess.TimeoutExpired:
        return {"ok": False, "consulta_espacial": False,
                "error": "la consulta superó 60 s"}
    except OSError as exc:
        return {"ok": False, "consulta_espacial": False,
                "error": f"no pude ejecutar el motor: {exc}"}

    if not cp.stdout.strip():
        return {"ok": False, "consulta_espacial": False,
                "error": "el motor no devolvió salida",
                "detalle": (cp.stderr or "")[-400:]}
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "consulta_espacial": False,
                "error": "salida del motor no es JSON",
                "detalle": cp.stdout[-400:]}
    data["ok"] = True
    data["consulta_espacial"] = bool(data.get("soportada"))
    data["generado"] = datetime.now(timezone.utc).isoformat()
    return data


def consulta_cached(q: str, region: list | None = None) -> dict:
    """Cache por (consulta, zona) invalidada por mtime de telemetría y máscaras."""
    sig = (_mtime_ns(TELEMETRY_CSV), _mtime_ns(ENTREGA_DIR / "masks"),
           _mtime_ns(_consulta_script()))
    key = (q, json.dumps(region, sort_keys=True) if region else None)
    if _CONSULTA_CACHE.get("key") == key and _CONSULTA_CACHE.get("sig") == sig:
        return _CONSULTA_CACHE["data"]
    data = run_consulta(q, region)
    _CONSULTA_CACHE.update({"key": key, "sig": sig, "data": data})
    return data


_CONSULTA_CACHE: dict = {"key": None, "sig": None, "data": None}


# --------------------------------------------------------------------------- #
# Grad-CAM (explicabilidad de los modelos de daño)
# --------------------------------------------------------------------------- #
# La herramienta vive en el repo de vuelo (tools/gradcam.py, hooks de torch)
# y se invoca por subproceso para que la estación siga stdlib-only. El PNG se
# cachea en outputs/gradcam/ y se sirve por /img/. Sin checkpoint o sin torch
# degrada con error claro y la web esconde el botón.

GRADCAM_MODELOS = {
    # clave API: (checkpoint en el repo de vuelo, nº de clases, arquitectura)
    "dano2": ("best_damage3_bal.pth", 3, "deeplabv3plus"),   # modelo de vuelo
    "dano": ("best_damage3.pth", 3, "deeplabv3plus"),        # xBD Joplin/Nepal
}
GRADCAM_FUENTES = ("full_res", "high_res", "thumb", "vis")
GRADCAM_TIMEOUT = 240
_GRADCAM_LOCK = threading.Lock()


def gradcam_disponible() -> dict:
    """Qué modelos se pueden explicar acá (tool + checkpoint presentes)."""
    script = FLIGHT_ROOT / "tools" / "gradcam.py"
    if not script.is_file():
        return {"ok": False, "motivo": "no encontré tools/gradcam.py en el repo de vuelo"}
    modelos = {}
    for key, (ckpt, _n, _arch) in GRADCAM_MODELOS.items():
        modelos[key] = (FLIGHT_ROOT / "outputs" / ckpt).is_file()
    return {"ok": any(modelos.values()), "modelos": modelos}


def _gradcam_fuente(src: str) -> Path | None:
    """Imagen fuente del frame, priorizando la resolución más alta."""
    for key in GRADCAM_FUENTES:
        for k, directory, suffix in BUCKETS:
            if k == key:
                found = find_image(directory, src, suffix)
                if found:
                    return found
    return None


def _gradcam_srcs() -> set[str]:
    payload = CACHE.get()
    return {str(fr.get("src")) for fr in payload.get("frames") or []}


def run_gradcam(src: str, modelo: str = "dano2",
                clase: int | None = None) -> dict:
    """Genera (o reutiliza) el Grad-CAM de un frame. Nunca lanza."""
    modelo = (modelo or "dano2").strip()
    if modelo not in GRADCAM_MODELOS:
        return {"ok": False, "error": f"modelo desconocido: {modelo}",
                "modelos": sorted(GRADCAM_MODELOS)}
    ckpt_name, n_cls, arch = GRADCAM_MODELOS[modelo]
    ckpt = FLIGHT_ROOT / "outputs" / ckpt_name
    if not ckpt.is_file():
        return {"ok": False, "error": f"falta el checkpoint {ckpt_name}",
                "detalle": f"esperado en {ckpt}"}
    if src not in _gradcam_srcs():
        return {"ok": False, "error": f"frame no encontrado: {src}"}
    img = _gradcam_fuente(src)
    if img is None:
        return {"ok": False, "error": f"el frame {src} no tiene imagen en disco"}

    out = GRADCAM_DIR / f"{src}_{modelo}.jpg"
    if out.is_file() and _mtime_ns(out) > max(_mtime_ns(img), _mtime_ns(ckpt)):
        return {"ok": True, "src": src, "modelo": modelo, "cached": True,
                "url": _rel(out), "fuente": _rel(img), "clase": clase}

    script = FLIGHT_ROOT / "tools" / "gradcam.py"
    if not script.is_file():
        return {"ok": False, "error": f"no encontré {script}"}
    cmd = [_consulta_python(), str(script),
           "--image", str(img), "--checkpoint", str(ckpt),
           "--arch", arch, "--num-classes", str(n_cls),
           "--img-size", "320", "--out", str(out)]
    if clase is not None:
        cmd += ["--clase", str(clase)]

    t0 = time.perf_counter()
    with _GRADCAM_LOCK:
        if out.is_file() and _mtime_ns(out) > max(_mtime_ns(img), _mtime_ns(ckpt)):
            pass  # otro hilo lo generó mientras se esperaba el lock
        else:
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", cwd=str(FLIGHT_ROOT),
                                    timeout=GRADCAM_TIMEOUT)
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"Grad-CAM superó {GRADCAM_TIMEOUT} s"}
            except OSError as exc:
                return {"ok": False, "error": f"no pude ejecutar Grad-CAM: {exc}"}
            if cp.returncode != 0 or not out.is_file():
                return {"ok": False, "error": "Grad-CAM falló",
                        "detalle": ((cp.stdout or "") + (cp.stderr or ""))[-400:]}
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"ok": True, "src": src, "modelo": modelo, "cached": False,
            "url": _rel(out), "fuente": _rel(img), "clase": clase,
            "ms": ms, "generado": datetime.now(timezone.utc).isoformat()}


def build_frame_payload(src: str) -> dict:
    """
    Una sola fila del CSV + sus imagenes disponibles.

    ARREGLADO: antes llamaba a ``read_telemetry()`` (re-lee y re-parsea el CSV
    entero, con Sniffer incluido) en CADA request de ``/api/frame/<src>``, sin
    pasar por la cache. Ahora reutiliza ``CACHE.get()``, que ya trae los frames
    con su mapa de ``files`` calculado.
    """
    src_norm = (src or "").strip()
    payload = CACHE.get()
    for fr in payload.get("frames") or []:
        if str(fr.get("src")) == src_norm:
            # Copia: el llamador no debe poder mutar lo que esta en la cache.
            return {"ok": True, "src": src_norm, "frame": dict(fr),
                    "summary": payload.get("summary")}
    return {"ok": False, "error": f"frame no encontrado: {src_norm}", "src": src_norm}


# --------------------------------------------------------------------------- #
# Handler HTTP
# --------------------------------------------------------------------------- #

def _safe_resolve(relpath: str) -> Path | None:
    """
    Resuelve una ruta relativa a ROOT rechazando cualquier escape (``..``,
    symlinks fuera del proyecto, rutas absolutas).
    """
    if not relpath:
        return None
    rel = unquote(relpath).replace("\\", "/")
    if rel.startswith("/"):
        rel = rel.lstrip("/")
    if "\x00" in rel:
        return None
    candidate = (ROOT / rel).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return candidate


def _safe_image_resolve(relpath: str) -> Path | None:
    """Resuelve solo imágenes publicables del contrato, nunca archivos arbitrarios."""
    candidate = _safe_resolve(relpath)
    if candidate is None or candidate.suffix.lower() not in IMG_EXTS:
        return None
    allowed = [CORRIDOR_MAP, GRADCAM_DIR, *(
        directory for _key, directory, _suffix in BUCKETS
    )]
    for base in allowed:
        try:
            if candidate == base.resolve() or candidate.relative_to(base.resolve()):
                return candidate
        except ValueError:
            continue
    return None


def static_root() -> Path:
    """Raíz de estáticos: el build de Vite si está presente, si no web/ legacy.
    Se resuelve por petición (un stat) para que un build nuevo se note al recargar."""
    return DIST_DIR if (DIST_DIR / "index.html").is_file() else WEB_DIR


def _safe_resolve_in(base: Path, relpath: str) -> Path | None:
    """Como _safe_resolve pero acotado a una raíz concreta (web/ o dist/)."""
    rel = unquote(relpath).replace("\\", "/").lstrip("/")
    if not rel or "\x00" in rel:
        return None
    cand = (base / rel).resolve()
    try:
        cand.relative_to(base.resolve())
    except ValueError:
        return None
    return cand


class GroundStationHandler(BaseHTTPRequestHandler):
    """Handler con enrutado simple: API JSON + estáticos + proxy de imágenes."""

    server_version = "CanSatGroundStation/1.0"
    protocol_version = "HTTP/1.1"

    # -- utilidades de respuesta ------------------------------------------- #

    def _send_bytes(self, status: int, body: bytes, ctype: str,
                    extra_headers: dict | None = None, no_cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # CORS: faltaba por completo. En dev no se notaba (el proxy de Vite
        # evita el cross-origin), pero la tablet de campo abriendo la app desde
        # otro host/puerto o `file://` quedaba bloqueada por el navegador. La
        # API es solo lectura (GET/HEAD), así que `*` no expone nada.
        self.send_header("Access-Control-Allow-Origin", "*")
        if no_cache:
            # Fundamental para el auto-refresh: el navegador no debe cachear
            # frames que el pipeline acaba de reescribir en disco.
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", no_cache=True)

    def _send_file(self, path: Path, ctype: str | None = None, no_cache: bool = False) -> None:
        if not path.is_file():
            self._send_json({"ok": False, "error": "no encontrado", "path": str(path)}, 404)
            return
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        try:
            data = path.read_bytes()
        except OSError as exc:
            self._send_json({"ok": False, "error": str(exc)}, 500)
            return
        self._send_bytes(200, data, ctype, no_cache=no_cache)

    # -- SSE: push de cambios de telemetría/resumen ------------------------- #

    @staticmethod
    def _watch_sig() -> str:
        a = TELEMETRY_CSV.stat().st_mtime if TELEMETRY_CSV.is_file() else 0.0
        b = SUMMARY_JSON.stat().st_mtime if SUMMARY_JSON.is_file() else 0.0
        return f"{a:.3f}:{b:.3f}"

    # Techo de vida de una conexion SSE. Sin el, un navegador que se cuelga sin
    # cerrar el socket deja el hilo vivo para siempre.
    SSE_MAX_SECONDS = 30 * 60
    SSE_POLL_SECONDS = 0.5
    SSE_BEAT_SECONDS = 15

    def _sse_events(self) -> None:
        """
        Server-Sent Events: emite ``data:`` cuando cambia el CSV o el summary.

        ARREGLADO: el loop anterior era ``while True`` SIN NINGUNA condicion de
        salida. Cuando el cliente se desconectaba, el ``write`` lanzaba
        ``BrokenPipeError`` que ``do_GET`` capturaba con ``pass`` -- el hilo
        salia, pero el ``EventSource`` del navegador reconecta cada 1.5 s
        (``retry: 1500``), asi que cada recarga o perdida de red dejaba hilos
        acumulados. En un vuelo de 20 minutos con el operador refrescando eran
        cientos de hilos vivos en la PC de ops.

        Ahora: (1) un heartbeat cada 15 s detecta la desconexion y sale del loop,
        y (2) hay un techo de vida por conexion, tras el cual se cierra y el
        navegador reconecta limpio.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        # Sin Content-Length: es un stream. X-Accel-Buffering por si hay proxy.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        t0 = time.monotonic()
        last = self._watch_sig()
        last_beat = -1
        try:
            self.wfile.write(b"retry: 1500\n\n")
            self.wfile.flush()
            while True:
                time.sleep(self.SSE_POLL_SECONDS)
                age = time.monotonic() - t0
                if age > self.SSE_MAX_SECONDS:
                    # Cierre deliberado: el EventSource reconecta solo.
                    self.wfile.write(b"event: rotate\ndata: reconnect\n\n")
                    self.wfile.flush()
                    break
                cur = self._watch_sig()
                if cur != last:
                    last = cur
                    self.wfile.write(f"data: {cur}\n\n".encode())
                    self.wfile.flush()
                elif int(age / self.SSE_BEAT_SECONDS) > last_beat:
                    # Heartbeat: comentario SSE (no dispara onmessage) que sirve
                    # de detector de desconexion -- si el cliente se fue, el
                    # write lanza y salimos.
                    last_beat = int(age / self.SSE_BEAT_SECONDS)
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass      # el cliente se fue: salida limpia del hilo

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        """Log compacto en una línea, con hora local."""
        sys.stderr.write("[ground-station] %s %s\n" % (
            datetime.now().strftime("%H:%M:%S"), fmt % args))

    # -- enrutado ----------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except BrokenPipeError:
            pass  # el navegador canceló la petición (cambio de vista): no es error
        except Exception:
            traceback.print_exc()
            try:
                self._send_json({"ok": False, "error": "error interno del servidor"}, 500)
            except Exception:
                pass

    def do_HEAD(self) -> None:  # noqa: N802
        """
        HEAD sin cuerpo.

        ARREGLADO: antes delegaba en ``do_GET()``, que esta bien para rutas
        normales (``_send_bytes`` ya omite el cuerpo cuando
        ``self.command == "HEAD"``) pero COLGABA PARA SIEMPRE en ``/api/events``,
        que es un stream infinito. Un health-check estandar (``curl -I``) dejaba
        un hilo bloqueado.
        """
        path = unquote(urlparse(self.path).path).rstrip("/")
        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.do_GET()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        # Allow-Origin faltaba acá también (solo estaban Methods/Headers).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _route(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.endswith("/") and path != "/":
            path = path.rstrip("/")

        # ---- API -------------------------------------------------------- #
        if path == "/api/health":
            self._send_json({
                "ok": True,
                "server": self.server_version,
                "time": datetime.now(timezone.utc).isoformat(),
                "telemetry_exists": TELEMETRY_CSV.is_file(),
                "summary_exists": SUMMARY_JSON.is_file(),
                "masks_exists": (ENTREGA_DIR / "masks").is_dir(),
                "detections_exists": TELEMETRY_JSONL.is_file(),
                "gradcam": gradcam_disponible(),
            })
            return

        if path == "/api/mission":
            self._send_json(CACHE.get())
            return

        if path == "/api/summary":
            summary = read_summary()
            self._send_json({"ok": summary is not None, "summary": summary})
            return

        if path == "/api/samples":
            # Campos por frame que no están en el CSV (incertidumbre, tiempos,
            # nodata) + agregados del panel de Muestreo.
            data = read_jsonl_extras()
            self._send_json({"ok": True, "samples": data["extras"],
                             "resumen": data["resumen"]})
            return

        if path == "/api/consulta":
            # Consulta Terrestre simbólica. GET con q=… y poly=[[lon,lat],…].
            qs = parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0]
            poly_raw = (qs.get("poly") or [""])[0]
            region = None
            if poly_raw:
                try:
                    region = json.loads(poly_raw)
                except json.JSONDecodeError:
                    self._send_json({"ok": False, "consulta_espacial": False,
                                     "error": "poly no es JSON válido"}, 400)
                    return
            self._send_json(consulta_cached(q, region))
            return

        if path == "/api/gradcam":
            # Explicabilidad del daño. GET con src=…&modelo=dano2|dano&clase=…
            qs = parse_qs(parsed.query)
            src = (qs.get("src") or [""])[0].strip()
            modelo = (qs.get("modelo") or ["dano2"])[0]
            clase_raw = (qs.get("clase") or [""])[0]
            clase = None
            if clase_raw:
                try:
                    clase = int(clase_raw)
                except ValueError:
                    self._send_json({"ok": False,
                                     "error": "clase debe ser entero"}, 400)
                    return
            if not src:
                self._send_json({"ok": False, "error": "falta src"}, 400)
                return
            # Siempre 200 con bandera ok (como /api/consulta): los errores de
            # input van en el JSON para no ensuciar la consola del jurado con
            # 404 de recursos que el frontend pide a propósito.
            data = run_gradcam(src, modelo, clase)
            self._send_json(data)
            return

        if path == "/api/events":
            self._sse_events()
            return

        if path.startswith("/api/frame/"):
            src = path[len("/api/frame/"):]
            payload = build_frame_payload(src)
            self._send_json(payload, 200 if payload.get("ok") else 404)
            return

        if path.startswith("/api/"):
            self._send_json({"ok": False, "error": f"endpoint desconocido: {path}"}, 404)
            return

        # ---- Proxy de imágenes desde la raíz del proyecto ---------------- #
        if path.startswith("/img/"):
            rel = path[len("/img/"):]
            target = _safe_image_resolve(rel)
            if target is None or not target.is_file():
                self._send_json({"ok": False, "error": "imagen no disponible",
                                 "path": rel}, 404)
                return
            self._send_file(target, no_cache=True)
            return

        # ---- Estáticos de la web ---------------------------------------- #
        if path in ("/favicon.ico", "/favicon.svg"):
            self._send_bytes(200, FAVICON_SVG, "image/svg+xml")
            return

        if path == "/" or path == "":
            self._send_file(static_root() / "index.html")
            return

        # Estáticos de la raíz activa (dist/ o web/), sin escapes de ruta.
        root = static_root()
        static = _safe_resolve_in(root, path)
        if static and static.is_file():
            # HTML/CSS/JS sin caché fuerte; los assets con hash pueden cachearse.
            no_cache = static.suffix in (".html", ".css", ".js") and "assets" not in static.parts
            self._send_file(static, no_cache=no_cache)
            return

        self._send_bytes(404, "<h1>404 — no encontrado</h1>".encode("utf-8"),
                         "text/html; charset=utf-8")


# --------------------------------------------------------------------------- #
# Arranque
# --------------------------------------------------------------------------- #

def banner(payload: dict) -> str:
    """Mensaje de bienvenida con el estado real de los artefactos."""
    frames = payload.get("n_frames", 0)
    alertas = sum(1 for f in payload.get("frames", []) if f.get("alert") == 1)
    high = sum(1 for f in payload.get("frames", []) if f.get("sample_pri") == "HIGH")
    lines = [
        "",
        "  [sat] ESTACION TERRENA - CanSat LB135",
        "  " + "-" * 52,
        f"  Telemetria : {'OK ' + str(frames) + ' frames' if frames else '-- sin telemetry.csv'}"
        f"   ({alertas} alertas / {high} HIGH)",
        f"  Post-vuelo : {'OK summary.json' if payload.get('summary') else '-- sin summary.json (seccion oculta)'}",
        f"  Corredor   : {'OK corridor_map.jpg' if payload['assets'].get('corridor_map') else '-- sin mapa'}",
        f"  Máscaras   : {'OK entrega/masks' if (ENTREGA_DIR / 'masks').is_dir() else '-- sin máscaras (consulta limitada)'}",
        f"  Estáticos  : {static_root().relative_to(ROOT)}",
        "  " + "-" * 52,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Estación terrena web · CanSat LB135")
    ap.add_argument("--host", default="127.0.0.1", help="interfaz a escuchar (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="puerto HTTP (default: 8000)")
    ap.add_argument("--root", default=None, help="raíz del proyecto (default: carpeta de este script)")
    ap.add_argument("--flight-root", default=None,
                    help="repo de vuelo para el motor de consultas "
                         "(default: ../cansat_seg_poc)")
    ap.add_argument("--consulta-python", default=None,
                    help="intérprete para tools/consulta.py "
                         "(default: venv del repo de vuelo, si no el actual)")
    args = ap.parse_args(argv)

    global FLIGHT_ROOT, CONSULTA_PYTHON
    if args.flight_root:
        FLIGHT_ROOT = Path(args.flight_root).resolve()
    if args.consulta_python:
        CONSULTA_PYTHON = args.consulta_python

    if args.root:
        # ARREGLADO: este bloque DUPLICABA a mano la definicion de todas las
        # rutas y reconstruia BUCKETS. Dos fuentes de verdad: si alguien agregaba
        # un bucket arriba, se olvidaba de aca y --root servia un contrato
        # distinto al de la raiz por defecto.
        _configure(Path(args.root))

    if not (WEB_DIR / "index.html").is_file():
        print(f"[!] Falta {WEB_DIR/'index.html'} — la web no podrá servirse.", file=sys.stderr)

    try:
        preview = CACHE.get()
    except Exception:
        preview = {"n_frames": 0, "frames": [], "summary": None, "assets": {"corridor_map": None}}
    print(banner(preview))

    host_show = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"  > Abriendo: http://{host_show}:{args.port}   (Ctrl+C para detener)\n")

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), GroundStationHandler)
    except OSError as exc:
        print(f"[!] No se pudo iniciar el servidor en {args.host}:{args.port}: {exc}",
              file=sys.stderr)
        print("    Probá con otro puerto:  python web_server.py --port 8080", file=sys.stderr)
        return 1

    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ground-station] detenido por el operador.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
