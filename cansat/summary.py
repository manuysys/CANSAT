"""
Schema canónico de ``entrega/summary.json`` — el contrato entre el pipeline de
post-vuelo y la estación terrena.

════════════════════════════════════════════════════════════════════════════
POR QUÉ EXISTE ESTE MÓDULO
════════════════════════════════════════════════════════════════════════════
Había **dos productores** del mismo archivo con schemas incompatibles:

  ``post_flight.py``                ``tools/make_demo_mission.py``
  ─────────────────────────         ────────────────────────────────
  "alertas": ["cap_0004", ...]      "alertas": [{"src":..., "diag":...,
              ↑ list[str]                        "danado_pct":..., "motivo":...}]
                                                 ↑ list[dict]
  "archivos": {"corredor": "...",   "archivos": {"telemetry": "...",
               "enhanced": [...],                 "corridor_map": "...",
               "evidencias": "...",               "vis": 12, "high_res": 12,
               "seg_b5": "..."}                   "ens_seg": 6, ...}
              ↑ rutas                              ↑ conteos enteros

Claves distintas para lo mismo (``corredor``/``corridor_map``,
``evidencias``/``vis``, ``seg_b5``/``ens_seg``) y tipos distintos.

El frontend **se rindió**: tipó ``alertas?: unknown[]`` y usó únicamente
``alerts.length``, descartando ``diag``, ``danado_pct``, ``motivo`` y
``sample_pri``. Como la estación terrena se prueba contra la demo, el camino
real (``post_flight.py``) nunca se ejercitó.

Este módulo es la definición única. Ambos productores llaman a
:func:`build_summary`; el consumidor valida con :func:`normalize_summary`, que
**además acepta los dos formatos legacy** para no romper un ``summary.json``
ya generado. El espejo TypeScript vive en
``web-app/src/lib/summarySchema.ts`` y debe mantenerse en sincronía
(hay un test que lo verifica).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Sequence

SCHEMA_VERSION: int = 2
MISION: str = "LB135"

# Buckets de imagen del contrato. Deben coincidir con BUCKETS en web_server.py
# y con FrameFiles en web-app/src/lib/types.ts.
BUCKETS: tuple = ("vis", "high_res", "full_res", "thumb", "ens_seg", "enhanced")


@dataclass
class Alerta:
    """Un evento de atención detectado a bordo."""

    src: str
    t_s: float | None = None
    alt_m: float | None = None
    diag: str | None = None
    danado_pct: float | None = None
    sample_pri: str | None = None
    motivo: str | None = None

    @staticmethod
    def from_row(row: dict[str, Any]) -> Alerta:
        """Construye desde una fila de ``telemetry.csv`` (str o ya tipada)."""
        diag = row.get("diag")
        dan = _num(row.get("danado_pct"))
        pri = row.get("sample_pri")
        motivo = row.get("motivo")
        if not motivo and diag:
            motivo = f"{diag} · daño {dan:.1f}%" if dan is not None else str(diag)
        return Alerta(
            src=str(row.get("src") or ""),
            t_s=_num(row.get("t_s")),
            alt_m=_num(row.get("alt_m")),
            diag=str(diag) if diag is not None else None,
            danado_pct=dan,
            sample_pri=str(pri) if pri else None,
            motivo=str(motivo) if motivo else None,
        )


@dataclass
class Archivos:
    """
    Artefactos del paquete de entrega.

    ``rutas`` mapea clave → ruta relativa al proyecto (o ``None``).
    ``conteos`` mapea bucket → cantidad de archivos (o ``None`` si el bucket no
    aplica). Antes un productor ponía rutas y el otro conteos en el MISMO campo;
    ahora van separados y ambos son opcionales.
    """

    rutas: dict[str, str | None] = field(default_factory=dict)
    conteos: dict[str, int | None] = field(default_factory=dict)


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def build_summary(
    rows: Sequence[dict[str, Any]],
    danado_por_frame: dict[str, float] | None = None,
    terrain_por_frame: dict[str, dict[str, float]] | None = None,
    rutas: dict[str, str | None] | None = None,
    conteos: dict[str, int | None] | None = None,
    nota: str | None = None,
    mision: str = MISION,
    supuestos_perdidas: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Arma el ``summary.json`` canónico a partir de las filas de telemetría.

    ``rows``: dicts con las columnas del CSV (``src``, ``alt_m``, ``verdict``,
    ``people``, ``vehicles``, ``alert``, ``danado_pct``, ``diag``, ``t_s``,
    ``sample_pri``, y opcionalmente ``veg``/``bui``/``wat``/``bare``/``oth``).
    """
    if not rows:
        raise ValueError("build_summary() recibió 0 filas de telemetría.")

    alts = [a for a in (_num(r.get("alt_m")) for r in rows) if a is not None]
    verd: dict[str, int] = {}
    for r in rows:
        k = str(r.get("verdict") or "SIN DATOS")
        verd[k] = verd.get(k, 0) + 1

    alertas = [asdict(Alerta.from_row(r)) for r in rows if int(_num(r.get("alert")) or 0) == 1]

    # src duplicados se pisan en los diccionarios por frame. Puede pasar si el
    # pipeline se corrió dos veces sobre la misma carpeta: mejor avisar.
    srcs = [str(r.get("src")) for r in rows]
    dupes = sorted({x for x in srcs if srcs.count(x) > 1})
    if dupes:
        print(
            f"  [WARN] summary: {len(dupes)} 'src' duplicado(s) en la telemetría "
            f"({', '.join(dupes[:4])}{'…' if len(dupes) > 4 else ''}). "
            f"Los valores por frame conservan el ÚLTIMO."
        )

    dan = dict(danado_por_frame or {}) or {
        str(r.get("src")): _num(r.get("danado_pct")) or 0.0 for r in rows
    }
    dan_vals = [v for v in dan.values() if v is not None]

    if terrain_por_frame is None:
        terrain_por_frame = {}
        for r in rows:
            src = str(r.get("src"))
            vals = {k: _num(r.get(k)) for k in ("veg", "bui", "wat", "bare", "oth")}
            if any(v is not None for v in vals.values()):
                terrain_por_frame[src] = {k: (v if v is not None else 0.0) for k, v in vals.items()}

    # ── Estimación de pérdidas humanas (DPD) ──────────────────────────── #
    # Se calcula desde las filas (danado_pct + area_m2 por frame). Si el CSV
    # no trae area_m2 (telemetría vieja), devuelve 0 frames con daño y la nota
    # correspondiente, en vez de inventar un número.
    from .casualties import Supuestos, estimar_mision

    sup = Supuestos(**(supuestos_perdidas or {}))
    perdidas = estimar_mision([dict(r) for r in rows], sup)

    return {
        "schema_version": SCHEMA_VERSION,
        "mision": mision,
        "n_frames": len(rows),
        "alt_max_m": max(alts) if alts else None,
        "alt_min_m": min(alts) if alts else None,
        "veredictos": verd,
        "personas_total": int(sum(_num(r.get("people")) or 0 for r in rows)),
        "vehiculos_total": int(sum(_num(r.get("vehicles")) or 0 for r in rows)),
        "alertas": alertas,
        "danado_pct_por_frame": dan,
        "danado_pct_prom": round(sum(dan_vals) / len(dan_vals), 2) if dan_vals else 0.0,
        "terrain_b5_por_frame": terrain_por_frame,
        "perdidas": perdidas,
        "archivos": {
            "rutas": dict(rutas or {}),
            "conteos": dict(conteos or {}),
        },
        "generado": datetime.now(timezone.utc).isoformat(),
        **({"nota": nota} if nota else {}),
    }


def normalize_summary(raw: dict[str, Any] | Any | None) -> dict[str, Any] | None:
    """
    Lleva un ``summary.json`` de CUALQUIER versión al schema v2.

    Acepta los dos formatos legacy de ``alertas`` (``list[str]`` y
    ``list[dict]``) y los dos de ``archivos`` (rutas planas y conteos planos).
    Devuelve ``None`` si la entrada no es un dict.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return {"schema_version": SCHEMA_VERSION, "raw": raw}

    out = dict(raw)

    # ── alertas: list[str] → list[dict] ─────────────────────────────────
    al = out.get("alertas")
    if isinstance(al, list):
        norm: list[dict[str, Any]] = []
        for a in al:
            if isinstance(a, dict) and "src" in a:
                norm.append(a)
            elif isinstance(a, str):
                norm.append(asdict(Alerta(src=a)))
            else:
                norm.append(asdict(Alerta(src=str(a))))
        out["alertas"] = norm

    # ── archivos: plano → {rutas, conteos} ──────────────────────────────
    arch = out.get("archivos")
    if isinstance(arch, dict) and "rutas" not in arch and "conteos" not in arch:
        rutas: dict[str, str | None] = {}
        conteos: dict[str, int | None] = {}
        # Renombres del formato legacy de post_flight.py → claves canónicas.
        alias = {
            "corredor": "corridor_map",
            "evidencias": "vis",
            "seg_b5": "ens_seg",
            "telemetria_csv": "telemetry",
        }
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

    out.setdefault("schema_version", 1)
    out.setdefault("mision", MISION)
    out.setdefault("veredictos", {})
    return out


def write_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    """Escribe el summary canónico creando los directorios padres."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def read_summary(path: str | Path) -> dict[str, Any] | None:
    """Lee y normaliza. Devuelve ``None`` si no existe o está corrupto."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return normalize_summary(data)


def validate(summary: dict[str, Any], n_frames_csv: int | None = None) -> list[str]:
    """
    Avisos humanos (lista vacía = todo bien). Espejo de ``lib/validate.ts``.

    No lanza: el objetivo es degradar con un mensaje, nunca romper la estación.
    """
    avisos: list[str] = []
    if not isinstance(summary, dict):
        return ["summary no es un objeto"]

    nf = summary.get("n_frames")
    if n_frames_csv is not None and isinstance(nf, int) and nf != n_frames_csv:
        avisos.append(f"summary.n_frames ({nf}) ≠ frames del CSV ({n_frames_csv})")

    verd = summary.get("veredictos")
    if isinstance(verd, dict) and isinstance(nf, int) and sum(verd.values()) != nf:
        avisos.append(f"veredictos suman {sum(verd.values())} pero n_frames={nf}")

    for a in summary.get("alertas") or []:
        if not isinstance(a, dict) or not a.get("src"):
            avisos.append("alerta sin 'src'")
            break

    arch = summary.get("archivos") or {}
    if isinstance(arch, dict) and "rutas" not in arch and "conteos" not in arch:
        avisos.append("archivos en formato legacy plano (no v2)")

    if summary.get("schema_version", 1) < SCHEMA_VERSION:
        avisos.append(
            f"schema_version {summary.get('schema_version', 1)} "
            f"< {SCHEMA_VERSION} (se normalizó al leer)"
        )

    dan = summary.get("danado_pct_prom")
    if dan is not None and not (0.0 <= float(dan) <= 100.0):
        avisos.append(f"danado_pct_prom fuera de 0-100: {dan}")

    return avisos
