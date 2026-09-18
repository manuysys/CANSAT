"""
CanSat La Base — Baseline real de la zona de lanzamiento.
1) Baja mosaico satelital RGB (Esri World Imagery) → outputs/baseline.png
2) (opcional --footprints) Baja huellas de edificios (OSM) → overlay + conteo

Uso:
    python baseline_tool.py --lat -34.6037 --lon -58.4173 --zoom 18
    python baseline_tool.py --lat -34.6037 --lon -58.4173 --zoom 18 --footprints
"""
import argparse
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")
OUT = Path("outputs")


def merc(lat):
    return math.asinh(math.tan(math.radians(lat)))


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - merc(lat) / math.pi) / 2.0 * n)
    return x, y


CACHE_DIR = Path("outputs/tile_cache")


def _fetch_tile(url: str, cache_key: str, retries: int = 3, timeout: int = 30):
    """
    Baja un tile con caché local y reintentos.

    ⚠ La versión anterior no cacheaba, no reintentaba y hacía
      ``mosaic[...] = img`` sin verificar: si ``cv2.imdecode`` devolvía ``None``
      (respuesta no-imagen, rate limit, red cortada) la asignación lanzaba y se
      perdía todo el mosaico. En el día del vuelo, sin red, esto no corre.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / f"{cache_key}.jpg"
    if local.is_file() and local.stat().st_size > 0:
        img = cv2.imread(str(local))
        if img is not None:
            return img

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url,
                                         headers={"User-Agent": "CanSatLB135/1.0"})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"la respuesta ({len(data)} bytes) no decodifica como imagen")
            local.write_bytes(data)
            return img
        except Exception as e:
            last_err = e
            if attempt < retries:
                import time as _t
                wait = 2 ** attempt
                print(f"    [retry {attempt}/{retries}] {type(e).__name__}: {e} "
                      f"— espero {wait} s")
                _t.sleep(wait)
    print(f"    [WARN] tile {cache_key} no disponible ({last_err}); queda negro.")
    return None


def fetch_mosaic(lat, lon, z, tiles=3, crop=640):
    cx, cy = tile_xy(lat, lon, z)
    k = tiles // 2
    mosaic = np.zeros((256 * tiles, 256 * tiles, 3), np.uint8)
    faltan = 0
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            url = ESRI.format(z=z, y=cy + dy, x=cx + dx)
            img = _fetch_tile(url, f"{z}_{cy + dy}_{cx + dx}")
            if img is None:
                faltan += 1
                continue
            y0, x0 = (dy + k) * 256, (dx + k) * 256
            h, w = img.shape[:2]
            mosaic[y0:y0 + min(h, 256), x0:x0 + min(w, 256)] = img[:256, :256]
    if faltan:
        print(f"  [WARN] {faltan} de {tiles * tiles} tiles no se pudieron bajar; "
              f"el baseline tiene zonas negras.")
        print("         ⚠ El modelo siamés va a marcar esas zonas como 'cambio'. "
              "No lo uses así.")
    if faltan == tiles * tiles:
        raise RuntimeError("No se pudo bajar NINGÚN tile. ¿Hay red? "
                           "¿Esri cambió la URL o aplica rate limit?")
    # posición del punto en el mosaico
    n = 2 ** z
    fx = (lon + 180.0) / 360.0 * n - cx + k
    fy = (1.0 - merc(lat) / math.pi) / 2.0 * n - cy + k
    px, py = int(fx * 256), int(fy * 256)
    half = crop // 2
    px, py = max(half, min(mosaic.shape[1] - half, px)), \
        max(half, min(mosaic.shape[0] - half, py))
    return mosaic[py - half:py + half, px - half:px + half]


def fetch_buildings(s, w, n_, e):
    q = f"[out:json][timeout:90];way[building]({s},{w},{n_},{e});out geom;"
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=q.encode(),
                                 headers={"User-Agent": "CanSatLB135/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=120))["elements"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--footprints", action="store_true")
    ap.add_argument("--out", default=str(OUT / "baseline.png"))
    args = ap.parse_args(argv)

    print("[1/2] Bajando mosaico satelital (Esri World Imagery)...")
    try:
        base = fetch_mosaic(args.lat, args.lon, args.zoom, crop=args.crop)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), base)
    print(f"[OK] {outp} ({base.shape[1]}x{base.shape[0]})")

    meta = {"lat": args.lat, "lon": args.lon, "zoom": args.zoom,
            "buildings_expected": None}

    if args.footprints:
        print("[2/2] Bajando huellas de edificios (OSM Overpass)...")
        # bounds aproximados del crop
        res = 156543.03 * math.cos(math.radians(args.lat)) / (2 ** args.zoom)
        half_m = res * args.crop / 2
        dlat = half_m / 111320.0
        dlon = half_m / (111320.0 * math.cos(math.radians(args.lat)))
        s, w = args.lat - dlat, args.lon - dlon
        n_, e = args.lat + dlat, args.lon + dlon
        els = fetch_buildings(s, w, n_, e)

        # rasterizar polígonos sobre el crop
        # (la versión anterior calculaba ys/xs con np.linspace y no los usaba:
        #  código muerto. El mapeo real es el lineal de abajo.)
        mask = np.zeros((args.crop, args.crop), np.uint8)
        count = 0
        for el in els:
            if "geometry" not in el:
                continue
            pts = []
            for nd in el["geometry"]:
                ix = int((nd["lon"] - w) / (e - w) * args.crop)
                iy = int((n_ - nd["lat"]) / (n_ - s) * args.crop)
                pts.append([ix, iy])
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
                count += 1
        overlay = base.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5
                             + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        cv2.imwrite(str(OUT / "baseline_footprints.png"), overlay)
        meta["buildings_expected"] = count
        print(f"[OK] {count} edificios esperados → "
              f"outputs/baseline_footprints.png")

    with open(OUT / "baseline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[OK] outputs/baseline_meta.json")
    print("\n[LISTO] El siamés usará outputs/baseline.png como 'pre'.")



    return 0
if __name__ == "__main__":
    raise SystemExit(main())
