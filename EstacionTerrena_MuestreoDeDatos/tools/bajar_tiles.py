"""
bajar_tiles.py - Tiles de mapa OFFLINE para la trayectoria GPS de la estación.

La estación corre sin internet (PC del laboratorio): en lugar de un basemap
online (Leaflet + OSM), se descargan los tiles UNA VEZ alrededor del área de
vuelo y se sirven como estáticos desde ``web-app/public/tiles/`` (Vite los
copia al build). El componente ``MapOffline.tsx`` los dibuja con proyección
Web Mercator; si un tile no está, degrada a fondo oscuro.

Uso:
    python tools/bajar_tiles.py                    # El Palomar, z13-18
    python tools/bajar_tiles.py --lat -31.4 --lon -64.2 --radio 0.02
    python tools/bajar_tiles.py --zooms 15 16 17

Fuente: OpenStreetMap estándar (https://tile.openstreetmap.org). Volumen
modesto (≈120 tiles, ~4 MB) con User-Agent identificable y throttling, acorde
a la política de uso de tiles de OSM (uso educativo, descarga única).

(c) 2026 - CanSat LB135 "La Base" - uso educativo.
"""
from __future__ import annotations

import argparse
import math
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "web-app" / "public" / "tiles"

# El Palomar (mismo origen que make_demo_mission.py / sim_uart.py).
LAT0, LON0 = -34.6075, -58.6126

# Radio (grados de latitud) por zoom: cubre la deriva del demo (~220 m), un
# vuelo real de ~1 km y el margen de 2 tiles por lado que dibuja MapOffline.
RADIOS = {13: 0.028, 14: 0.020, 15: 0.016, 16: 0.014, 17: 0.010, 18: 0.008}

URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
UA = "CanSat-LB135-Educational/1.0 (+https://github.com/manuysys/CANSAT)"


def lonlat_a_tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Web Mercator: lat/lon -> coordenadas fraccionarias de tile (x, y)."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tiles_del_area(lat: float, lon: float, radio: float, z: int):
    x0f, y0f = lonlat_a_tile(lat + radio, lon - radio, z)  # noroeste
    x1f, y1f = lonlat_a_tile(lat - radio, lon + radio, z)  # sudeste
    for x in range(int(math.floor(x0f)), int(math.floor(x1f)) + 1):
        for y in range(int(math.floor(y0f)), int(math.floor(y1f)) + 1):
            yield x, y


def bajar(z: int, x: int, y: int, dest_dir: Path, timeout: float = 20.0) -> int:
    dest = dest_dir / str(z) / str(x) / f"{y}.png"
    if dest.is_file() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL.format(z=z, x=x, y=y),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=LAT0)
    ap.add_argument("--lon", type=float, default=LON0)
    ap.add_argument("--radio", type=float, default=None,
                    help="radio en grados (pisa la tabla por zoom)")
    ap.add_argument("--zooms", nargs="*", type=int, default=sorted(RADIOS),
                    help="zooms a bajar (default 13..18)")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    dest_dir: Path = args.dest
    total = bajados = errores = 0
    bytes_total = 0
    for z in args.zooms:
        radio = args.radio if args.radio is not None else RADIOS.get(z, 0.01)
        lista = list(tiles_del_area(args.lat, args.lon, radio, z))
        n_ok = 0
        for x, y in lista:
            total += 1
            try:
                n = bajar(z, x, y, dest_dir)
                bytes_total += n
                if n:
                    bajados += 1
                n_ok += 1
                if n:
                    time.sleep(0.08)  # cortesía con el servidor de tiles
            except Exception as e:
                errores += 1
                print(f"  [WARN] z{z}/{x}/{y}: {type(e).__name__}: {e}")
        print(f"  z{z}: {n_ok}/{len(lista)} tiles (radio {radio}°)")
    mb = bytes_total / 1e6
    print(f"[OK] {bajados} tiles nuevos ({total} en total) · {mb:.2f} MB "
          f"· {errores} errores · destino {dest_dir.relative_to(ROOT)}")
    return 0 if errores == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
