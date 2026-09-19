"""
Densidad poblacional por coordenada — CanSat LB135.

El DPD pide estimar las pérdidas humanas; `cansat/casualties.py` usa una
densidad de población que hasta ahora era un supuesto fijo (1500 hab/km²).
Este módulo la reemplaza por el **dato real** cuando hay GPS: lee una grilla
gruesa (0.1°) construida desde WorldPop 1 km con
``tools/build_pop_grid.py`` y hace un lookup por frame.

- Sin grilla o sin fix GPS → devuelve el valor por defecto (y la telemetría
  declara la fuente). Nunca lanza.
- La grilla es un CSV chico (`dataset/population/population_grid.csv`), así que
  corre en la Pi sin rasterio ni GDAL.

Atribución: WorldPop (CC BY 4.0) — ver el encabezado del builder.
"""

from __future__ import annotations

import csv
from pathlib import Path

GRID_DEFAULT = Path("dataset/population/population_grid.csv")
POP_DENSITY_FALLBACK: float = 1500.0

_cache: dict[Path, dict] = {}


def _clave(lat: float, lon: float) -> tuple[int, int]:
    """Celda de 0.1°: se redondea al centro más cercano."""
    return (round(lat * 10.0), round(lon * 10.0))


def cargar(path: str | Path = GRID_DEFAULT) -> dict:
    """
    Carga la grilla en un dict ``{(lat_idx, lon_idx): dens}`` (cache por ruta).

    Si el archivo no existe devuelve ``{}`` — el pipeline sigue con el default.
    """
    p = Path(path)
    if p in _cache:
        return _cache[p]
    grid: dict = {}
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    lat = float(r["lat"])
                    lon = float(r["lon"])
                    dens = float(r["dens_km2"])
                except (KeyError, ValueError):
                    continue
                grid[_clave(lat, lon)] = dens
    _cache[p] = grid
    return grid


def densidad(lat: float | None, lon: float | None,
             default: float = POP_DENSITY_FALLBACK,
             path: str | Path = GRID_DEFAULT) -> tuple[float, str]:
    """
    Densidad (hab/km²) para una posición. Devuelve ``(densidad, fuente)``.

    ``fuente`` es ``"worldpop"`` si la celda existe y ``"supuesto"`` si se usó
    el valor por defecto (sin GPS, sin grilla o celda vacía).
    """
    if lat is None or lon is None or (abs(lat) < 1e-9 and abs(lon) < 1e-9):
        return float(default), "supuesto"
    grid = cargar(path)
    if not grid:
        return float(default), "supuesto"
    dens = grid.get(_clave(float(lat), float(lon)))
    if dens is None:
        return float(default), "supuesto"
    return float(dens), "worldpop"
