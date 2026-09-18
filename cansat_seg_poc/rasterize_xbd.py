"""
CanSat La Base — Rasterizador masivo de xBD (fase D1, v3).
Parsea WKT con shapely y convierte todos los JSON post en máscaras de daño (0-4).
Salida: dataset/xbd_masks/*.png + manifest.csv

Uso:  python rasterize_xbd.py
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import shapely.wkt as swkt

XBD = Path("dataset/xbd")
OUT = Path("dataset/xbd_masks")

DAMAGE_MAP = {
    "no-damage": 1, "minor-damage": 2, "major-damage": 3,
    "destroyed": 4, "building": 1,
}


def level_of(props):
    for k in ("subtype", "damage", "xdamage", "damage_grade"):
        v = str(props.get(k, "")).lower()
        if v in DAMAGE_MAP:
            return DAMAGE_MAP[v]
    return 1


def collect_features(node, out):
    if isinstance(node, dict):
        if "properties" in node or "wkt" in node or "geometry" in node:
            out.append(node)
        else:
            for v in node.values():
                collect_features(v, out)
    elif isinstance(node, list):
        for v in node:
            collect_features(v, out)


def rings_of_feature(f):
    s = f.get("wkt") or ""
    if not s:
        return []
    try:
        geom = swkt.loads(s)
    except Exception:
        return []
    rings = []
    if geom.geom_type == "Polygon":
        rings.append(list(geom.exterior.coords))
    elif geom.geom_type == "MultiPolygon":
        rings += [list(p.exterior.coords) for p in geom.geoms]
    return rings


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jsons = sorted(XBD.rglob("*_post_disaster.json"))
    print(f"JSONs post encontrados: {len(jsons)}")

    rows = []
    for i, jp in enumerate(jsons):
        img_path = None
        for cand in [jp.with_suffix(".png"),
                     Path(str(jp).replace("labels", "images").replace(".json", ".png"))]:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        data = json.load(open(jp, encoding="utf-8"))
        feats = []
        collect_features(data.get("features", []), feats)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for f in feats:
            lvl = level_of(f.get("properties", {}))
            for ring in rings_of_feature(f):
                ip = np.array(ring, dtype=np.float32).round().astype(np.int32)
                if len(ip) >= 3:
                    cv2.fillPoly(mask, [ip], int(lvl))

        counts = [int((mask == c).sum()) for c in range(1, 5)]
        if sum(counts) == 0:
            continue

        cv2.imwrite(str(OUT / f"{jp.stem}.png"), mask)
        rows.append([jp.stem, str(img_path), str(OUT / f"{jp.stem}.png"), *counts])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(jsons)} procesados...")

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["name", "image", "mask", "intacto", "menor", "mayor", "destruido"])
        wtr.writerows(rows)
    print(f"[OK] {len(rows)} máscaras rasterizadas en {OUT.resolve()}")


if __name__ == "__main__":
    main()
