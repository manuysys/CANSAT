"""
Construye la grilla de densidad poblacional (WorldPop) — CanSat LB135.

Toma el GeoTIFF 1 km de WorldPop (p. ej. Argentina 2020 UN-adjusted, CC BY 4.0)
y lo agrega a una grilla gruesa (0.1° ≈ 11 km) con la densidad en hab/km².
Salida: `dataset/population/population_grid.csv` (lat,lon,dens) — chica, para
que `cansat/population.py` la lea en la Pi sin rasterio.

El GeoTIFF trae la georreferencia en los tags 33550 (escala de píxel) y 33922
(punto de amarre), que PIL expone: no hace falta GDAL.

Uso:
    curl -L -o dataset/population/arg_ppp_2020_1km_Aggregated_UNadj.tif \\
      https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/ARG/arg_ppp_2020_1km_Aggregated_UNadj.tif
    python tools/build_pop_grid.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cansat                                                       # noqa: E402,F401
import numpy as np                                                  # noqa: E402
from PIL import Image                                               # noqa: E402

NODATA = -99999.0
KM_POR_GRADO = 111.32


def leer_geotiff(path: Path) -> tuple[np.ndarray, float, float, float]:
    """Devuelve ``(array, escala_grados, lon0, lat0)`` del GeoTIFF de WorldPop."""
    im = Image.open(path)
    scale = float(im.tag_v2[33550][0])
    _i, _j, _k, lon0, lat0, _z = im.tag_v2[33922]
    arr = np.array(im, dtype=np.float32)
    arr[arr <= NODATA + 1] = 0.0
    return arr, scale, float(lon0), float(lat0)


def agregar(arr: np.ndarray, scale: float, lon0: float, lat0: float,
            cell: float = 0.1, min_density: float = 0.5) -> list[tuple[float, float, float]]:
    """
    Agrega el raster fino a celdas de ``cell`` grados y devuelve
    ``[(lat_centro, lon_centro, hab_km2), ...]`` con densidad > ``min_density``.
    """
    paso = max(1, round(cell / scale))               # píxeles finos por celda
    h, w = arr.shape
    hp = int(np.ceil(h / paso)) * paso
    wp = int(np.ceil(w / paso)) * paso
    pad = np.zeros((hp, wp), dtype=np.float64)
    pad[:h, :w] = arr

    bloques = pad.reshape(hp // paso, paso, wp // paso, paso).sum(axis=(1, 3))
    filas: list[tuple[float, float, float]] = []
    for i in range(bloques.shape[0]):
        lat_c = lat0 - (i * paso + paso / 2.0) * scale
        if not -90.0 <= lat_c <= 90.0:
            continue
        area_km2 = (KM_POR_GRADO * cell) ** 2 * max(0.05, np.cos(np.radians(lat_c)))
        for j in range(bloques.shape[1]):
            hab = float(bloques[i, j])
            if hab <= 0:
                continue
            dens = hab / area_km2
            if dens < min_density:
                continue
            lon_c = lon0 + (j * paso + paso / 2.0) * scale
            filas.append((round(float(lat_c), 4), round(float(lon_c), 4),
                          round(float(dens), 1)))
    return filas


def main() -> int:
    ap = argparse.ArgumentParser(description="WorldPop GeoTIFF → grilla 0.1°")
    ap.add_argument("--tif", default="dataset/population/arg_ppp_2020_1km_Aggregated_UNadj.tif")
    ap.add_argument("--out", default="dataset/population/population_grid.csv")
    ap.add_argument("--cell", type=float, default=0.1)
    ap.add_argument("--min-density", type=float, default=0.5)
    args = ap.parse_args()

    tif = Path(args.tif)
    if not tif.is_file():
        print(f"[ERROR] no existe {tif}")
        print("        Bajalo con el comando del docstring (17 MB, CC BY 4.0).")
        return 1

    arr, scale, lon0, lat0 = leer_geotiff(tif)
    print(f"raster {arr.shape} · pixel {scale:.6f}° · origen ({lat0:.3f}, {lon0:.3f})")
    filas = agregar(arr, scale, lon0, lat0, args.cell, args.min_density)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "dens_km2"])
        w.writerows(filas)

    total_hab = sum(r[2] for r in filas)
    dens_max = max((r for r in filas), key=lambda r: r[2])
    print(f"[OK] {len(filas)} celdas → {out}")
    print(f"     densidad máx {dens_max[2]:.0f} hab/km² en ({dens_max[0]}, {dens_max[1]})")
    print(f"     (las celdas vacías no se guardan: {total_hab:.0f} 'hab/km² sumados')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
