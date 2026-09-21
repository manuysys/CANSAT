"""
Detección de fuera-de-distribución (OOD) por frame — proxy DECLARADO.

No es un detector formal de novedad: compara lo que el pipeline ya mide de cada
frame contra una **referencia** calculada sobre LoveDA Val
(``tools/build_ood_reference.py`` → ``docs/benchmarks/ood_loveda_val.json``):

  · mezcla de clases del terreno (Jensen-Shannon vs la media de la referencia),
  · ``veg_exg_pct`` (ExG, vegetación por color) y ``shadow_pct`` (sombras),
    en z-scores contra la media/σ de la referencia.

Con eso la estación puede avisar "el modelo está fuera de su dominio" cuando el
vuelo cae en una escena que no se parece a la de entrenamiento (desierto, nieve,
costa, noche), que es exactamente donde los números de Val no aplican. El aviso
NO bloquea nada ni cambia decisiones: es una advertencia de alcance.

El umbral se declara en la referencia (p95 de la propia JS dentro de Val y
``UMBRAL_Z`` para los índices) y la respuesta incluye los componentes para
auditar por qué se marcó.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REFERENCIA_DEFECTO = Path("docs/benchmarks/ood_loveda_val.json")

#: z-score máximo tolerado para los índices de color (3σ).
UMBRAL_Z: float = 3.0


def js_divergence(p, q) -> float:
    """
    Divergencia de Jensen-Shannon en bits (base 2) entre dos distribuciones.

    Simétrica, en [0, 1]; 0 = idénticas. Tolera ceros y normaliza.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"distribuciones de distinto tamaño: {p.shape} vs {q.shape}")
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def cargar_referencia(path: str | Path = REFERENCIA_DEFECTO) -> dict | None:
    """Lee el JSON de referencia. ``None`` si no existe o está corrupto."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def score(
    pcts,
    veg_exg_pct: float | None = None,
    shadow_pct: float | None = None,
    referencia: dict | None = None,
) -> dict:
    """
    Score OOD del frame. ``ood_score`` es 1.0 = justo en el umbral declarado.

    Sin referencia devuelve todo ``None`` (no se inventa un número).
    """
    if not referencia:
        return {"ood_score": None, "ood_flag": None, "referencia": None,
                "componentes": {}}

    componentes: dict = {}
    clases_prom = referencia.get("clases_prom") or []
    if len(clases_prom) == len(pcts):
        js = js_divergence(pcts, clases_prom)
        umbral_js = float(referencia.get("umbral_js") or 0.0)
        componentes["js"] = round(js, 4)
        componentes["umbral_js"] = umbral_js
    else:
        js, umbral_js = 0.0, 0.0
        componentes["js"] = None

    z_max = 0.0
    for nombre, valor in (("veg_exg_pct", veg_exg_pct), ("shadow_pct", shadow_pct)):
        med = referencia.get(f"{nombre}_media")
        std = referencia.get(f"{nombre}_std")
        if valor is None or med is None or not std:
            continue
        z = abs(float(valor) - float(med)) / max(float(std), 1e-6)
        componentes[f"z_{nombre}"] = round(z, 2)
        z_max = max(z_max, z)

    ratio_js = js / umbral_js if umbral_js > 0 else 0.0
    score_val = max(ratio_js, z_max / UMBRAL_Z)
    return {
        "ood_score": round(float(score_val), 3),
        "ood_flag": bool(score_val > 1.0),
        "referencia": referencia.get("fuente"),
        "componentes": componentes,
    }
