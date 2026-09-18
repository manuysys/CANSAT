"""
Rutas del proyecto resueltas desde la ubicación del archivo, no desde el CWD.

Los 64 scripts de la raíz usan rutas relativas (``"outputs/baseline.png"``), así
que **sólo funcionan si se los invoca desde la raíz del proyecto**. Este módulo
da una resolución absoluta para que un script pueda ejecutarse desde cualquier
lado, y centraliza las rutas que estaban repetidas con valores distintos.
"""

from __future__ import annotations

from pathlib import Path

# Raíz del proyecto = padre del paquete ``cansat/``.
ROOT: Path = Path(__file__).resolve().parent.parent

OUTPUTS: Path = ROOT / "outputs"
MODELS: Path = ROOT / "models"
DATASET: Path = ROOT / "dataset"
ENTREGA: Path = ROOT / "entrega"
DOCS: Path = ROOT / "docs"
RUNS: Path = ROOT / "runs"
PI: Path = ROOT / "pi"

MISSION_DIR: Path = OUTPUTS / "mission"
TELEMETRY_CSV: Path = MISSION_DIR / "telemetry.csv"
TELEMETRY_JSONL: Path = MISSION_DIR / "telemetry.jsonl"
CORRIDOR_MAP: Path = OUTPUTS / "corridor_map.jpg"
SUMMARY_JSON: Path = ENTREGA / "summary.json"
BASELINE_PNG: Path = OUTPUTS / "baseline.png"
METRICS_DIR: Path = OUTPUTS / "metrics"


def p(*parts: str) -> Path:
    """Resuelve una ruta relativa contra la raíz del proyecto."""
    path = Path(*parts)
    return path if path.is_absolute() else (ROOT / path)


def require(path: str | Path, hint: str = "") -> Path:
    """
    Devuelve la ``Path`` absoluta o lanza ``FileNotFoundError`` con un mensaje
    accionable, en vez de dejar que reviente tres llamadas más tarde con ``None``.
    """
    q = p(path)
    if not q.exists():
        msg = f"No existe: {q}"
        raise FileNotFoundError(msg + (f"\n  → {hint}" if hint else ""))
    return q


def model_path(name: str, required: bool = True, hint: str = "") -> Path | None:
    """
    Resuelve un artefacto de ``outputs/`` o ``models/``.

    ``required=False`` devuelve ``None`` si falta (para modelos opcionales como
    el siamés o el flood specialist), pero **imprime un warning**: el bug
    original era que ``post_flight.py`` se salteaba un bloque entero en silencio
    porque el ONNX no existía.
    """
    for base in (OUTPUTS, MODELS, ROOT):
        cand = base / name
        if cand.is_file():
            return cand
    if required:
        raise FileNotFoundError(
            f"No encuentro el modelo '{name}' ni en {OUTPUTS} ni en {MODELS}.\n"
            f"  → {hint or 'Regeneralo con el script de export (ver MODELS.yaml).'}"
        )
    print(f"  [WARN] modelo opcional ausente: {name} — queda desactivado.")
    return None


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        p(str(d)).mkdir(parents=True, exist_ok=True)
