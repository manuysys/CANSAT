"""
CanSat La Base — Inspección de xBD (fase D1, v5).
Este mirror guarda la geometría como texto WKT (campo "wkt"); se parsea con shapely.
Uso:  python inspect_xbd.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import shapely.wkt as swkt

XBD = Path("dataset/xbd")
if not list(XBD.rglob("*.json")):
    XBD = Path("dataset/xbd_zip")

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
    jsons = sorted(XBD.rglob("*_post_disaster.json"),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not jsons:
        print("[ERROR] No se encontraron JSON de xBD.")
        raise SystemExit(1)

    jp = jsons[0]
    data = json.load(open(jp, encoding="utf-8"))
    feats = []
    collect_features(data.get("features", []), feats)
    print(f"JSON: {jp.name} | n features: {len(feats)}")
    print(f"  wkt ejemplo    : {str(feats[0].get('wkt'))[:100]}")

    img_path = None
    for cand in [jp.with_suffix(".png"),
                 Path(str(jp).replace("labels", "images").replace(".json", ".png"))]:
        if cand.exists():
            img_path = cand
            break
    if img_path is None:
        print("  [WARN] No encontré la imagen post.")
        return

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    n_rings = 0
    allvals = []
    for f in feats:
        lvl = level_of(f.get("properties", {}))
        for ring in rings_of_feature(f):
            pts = np.array(ring, dtype=np.float32)
            allvals.append(pts)
            ip = pts.round().astype(np.int32)
            if len(ip) >= 3:
                cv2.fillPoly(mask, [ip], int(lvl))
            n_rings += 1

    print(f"  anillos encontrados: {n_rings}")
    if allvals:
        allpts = np.concatenate(allvals)
        print(f"  rango X: {allpts[:, 0].min():.1f}..{allpts[:, 0].max():.1f}  "
              f"Y: {allpts[:, 1].min():.1f}..{allpts[:, 1].max():.1f}")

    out = Path("outputs/inspect_xbd_mask.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), mask)
    print(f"  imagen post    : {img_path.name} ({w}x{h})")
    print(f"  niveles en mask: {np.unique(mask).tolist()}  (0=fondo,1=intacto..4=destruido)")


if __name__ == "__main__":
    main()
