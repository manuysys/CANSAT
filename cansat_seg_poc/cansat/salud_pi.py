"""
Salud de la Raspberry Pi durante el vuelo (U2).

Lee temperatura (``/sys/class/thermal/thermal_zone0/temp``), uptime
(``/proc/uptime``) y throttling (``vcgencmd get_throttled``). Fuera de la Pi
devuelve ``None`` sin romper nada: las rutas y el runner son inyectables para
testear en la PC (los tests usan archivos temporales, no la placa).
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
UPTIME_PATH = Path("/proc/uptime")

#: Bits del registro de ``vcgencmd get_throttled`` (0x0 = todo bien).
#: Los 0-3 son el estado ACTUAL; los 16-19, "ocurrió desde el arranque".
FLAGS: dict[int, str] = {
    0: "subvoltaje_actual",
    1: "freq_capeada_actual",
    2: "throttled_actual",
    3: "limite_termico_suave_actual",
    16: "subvoltaje_ocurrio",
    17: "freq_capeada_ocurrio",
    18: "throttled_ocurrio",
    19: "limite_termico_suave_ocurrio",
}


def leer_temp_c(path: Path = TEMP_PATH) -> float | None:
    """Temperatura del SoC en °C (None si no está disponible)."""
    try:
        return int(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def leer_uptime_s(path: Path = UPTIME_PATH) -> float | None:
    """Uptime en segundos (None si no está disponible)."""
    try:
        return float(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def parse_throttled(texto: str) -> dict[str, bool] | None:
    """Parsea ``throttled=0x50000`` → flags booleanos por nombre."""
    try:
        valor = int(texto.strip().split("=")[1], 16)
    except (IndexError, ValueError):
        return None
    return {nombre: bool(valor & (1 << bit)) for bit, nombre in FLAGS.items()}


def leer_throttled(
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, bool] | None:
    """Estado de throttling vía ``vcgencmd`` (None si no está disponible)."""
    try:
        r = runner(["vcgencmd", "get_throttled"], capture_output=True,
                   text=True, timeout=5)
        return parse_throttled(r.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return None


def agregar(muestras: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Resumen para ``pi_health.json`` a partir de las muestras del vuelo.

    ``muestras``: dicts con ``temp_c`` (opcional), ``throttled`` (opcional) y
    ``uptime_s`` (opcional). Los flags de throttling se unen con OR lógico.
    """
    temps = [m["temp_c"] for m in muestras if m.get("temp_c") is not None]
    uptimes = [m["uptime_s"] for m in muestras if m.get("uptime_s") is not None]
    flags: dict[str, bool] = {}
    for m in muestras:
        for nombre, activo in (m.get("throttled") or {}).items():
            flags[nombre] = flags.get(nombre, False) or bool(activo)
    return {
        "n_muestras": len(muestras),
        "temp_c_min": round(min(temps), 1) if temps else None,
        "temp_c_max": round(max(temps), 1) if temps else None,
        "temp_c_prom": round(sum(temps) / len(temps), 1) if temps else None,
        "uptime_inicio_s": (round(min(uptimes), 1) if uptimes else None),
        "uptime_fin_s": (round(max(uptimes), 1) if uptimes else None),
        "throttled": flags or None,
        "throttled_alguna_vez": (any(flags.values()) if flags else None),
    }
