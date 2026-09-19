"""
CanSat La Base — F2 v3: prepara CRASAR-U-DROIDs (sUAS multi-desastre) para daño.

CRASAR (HF `CRASAR/CRASAR-U-DROIDs`, CC BY 4.0) trae ortomosaicos sUAS
(`*.geo.tif`) + anotaciones JSON por edificio con la Joint Damage Scale:
    no damage / minor damage / major damage / destroyed / un-classified / obscured

Este script corta cada ortomosaico en tiles de `--tile` px (lectura por
ventanas con `tifffile.memmap`, sin cargar 1 Gpx en RAM) y rasteriza los
polígonos a nuestro esquema de severidad de 5 clases:
    0=other  1=intacto  2=menor  3=mayor  4=destruido
(`un-classified` y `obscured` se ignoran).

Salida: `dataset/crasar_tiles/{split}/{images,masks}` + `manifest_{split}.csv`
con el formato de RescueNet (name,image,mask,intacto,menor,mayor,destruido).

Uso:
    python prepare_crasar.py --splits train test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cansat                                                       # noqa: E402,F401
import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import rasterio                                                     # noqa: E402
from rasterio.windows import Window                                 # noqa: E402

CRASAR = ROOT / "dataset/crasar"
DST = ROOT / "dataset/crasar_tiles"
JDS = {"no damage": 1, "minor damage": 2, "major damage": 3, "destroyed": 4}
MIN_EDIF_PX = 400          # mínimo de píxeles de edificio para guardar el tile


def leer_anotaciones(split: str, tif_name: str) -> list[dict]:
    """Entradas BDA (label + pixels) del JSON del ortomosaico."""
    p = (CRASAR / split / "annotations/UAS/building_damage_assessment"
         / f"{tif_name}.json")
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [e for e in data if e.get("label") in JDS and e.get("pixels")]


def main() -> int:
    ap = argparse.ArgumentParser(description="CRASAR → tiles 5 clases + manifest")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--jpeg", type=int, default=90)
    ap.add_argument("--limit-tif", type=int, default=0, help="solo N tif por split")
    args = ap.parse_args()

    for split in args.splits:
        img_dir = CRASAR / split / "imagery/UAS"
        if not img_dir.is_dir():
            print(f"[WARN] falta {img_dir}")
            continue
        out_img = DST / split / "images"
        out_msk = DST / split / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        tifs = sorted(img_dir.glob("*.tif"))
        if args.limit_tif:
            tifs = tifs[:args.limit_tif]
        rows: list[dict] = []
        for tif in tifs:
            ents = leer_anotaciones(split, tif.name)
            if not ents:
                print(f"  [i] {tif.name}: sin anotaciones BDA, se saltea")
                continue
            try:
                src = rasterio.open(tif)
            except Exception as e:
                print(f"  [WARN] {tif.name}: {type(e).__name__}: {e}")
                continue
            h, w = src.height, src.width
            # Índice espacial de polígonos por bounding box (para no recorrer
            # todos los polígonos en cada ventana).
            polys = []
            for e in ents:
                pts = np.array([[p["x"], p["y"]] for p in e["pixels"]],
                               dtype=np.int32)
                if len(pts) < 3:
                    continue
                polys.append((pts, JDS[e["label"]]))
            n_tiles = 0
            for y0 in range(0, max(1, h - args.tile + 1), args.tile):
                for x0 in range(0, max(1, w - args.tile + 1), args.tile):
                    win = np.array([[x0, y0], [x0 + args.tile, y0 + args.tile]])
                    cerca = [(p, c) for p, c in polys
                             if (p[:, 0].max() >= win[0, 0] and p[:, 0].min() <= win[1, 0]
                                 and p[:, 1].max() >= win[0, 1] and p[:, 1].min() <= win[1, 1])]
                    if not cerca:
                        continue
                    msk = np.zeros((args.tile, args.tile), dtype=np.uint8)
                    for p, c in cerca:
                        cv2.fillPoly(msk, [p - [x0, y0]], int(c))
                    n_edif = int((msk >= 1).sum())
                    if n_edif < MIN_EDIF_PX:
                        continue
                    stem = f"crasar_{split}_{tif.stem}_{y0}_{x0}"
                    arr = src.read(window=Window(x0, y0, args.tile, args.tile))
                    tile = np.transpose(arr[:3], (1, 2, 0))
                    if arr.shape[0] >= 3:
                        tile = tile[:, :, ::-1]          # RGB → BGR (cv2)
                    cv2.imwrite(str(out_img / f"{stem}.jpg"), tile,
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
                    cv2.imwrite(str(out_msk / f"{stem}.png"), msk)
                    rows.append({
                        "name": stem,
                        "image": str(out_img / f"{stem}.jpg"),
                        "mask": str(out_msk / f"{stem}.png"),
                        "intacto": int((msk == 1).sum()),
                        "menor": int((msk == 2).sum()),
                        "mayor": int((msk == 3).sum()),
                        "destruido": int((msk == 4).sum()),
                    })
                    n_tiles += 1
            src.close()
            print(f"  {tif.name}: {n_tiles} tiles")

        man = DST / f"manifest_{split}.csv"
        with man.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "image", "mask",
                                              "intacto", "menor", "mayor",
                                              "destruido"])
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {split}: {len(rows)} tiles → {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
