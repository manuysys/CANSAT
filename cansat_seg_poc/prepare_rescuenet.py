"""
CanSat La Base — F2: prepara RescueNet para el modelo de daño (óptico UAV).

RescueNet (Rahnemoonfar et al., Hurricane Michael, UAV 3000×4000) trae 10
clases. Se remapea a las 3 del modelo de vuelo:

    other (0)   = {0 fondo, 1 agua, 6 vehículo, 7 ruta-libre, 8 ruta-bloqueada,
                   9 árbol, 10 pileta}
    intacto (1) = {2 edificio sin daño}
    danado (2)  = {3 menor, 4 mayor, 5 destrucción total}

Genera tiles de ``--tile`` px (con cobertura del borde), filtra los que no
tienen construcción suficiente y escribe un manifest con el MISMO formato que
``dataset/xbd_masks/manifest.csv`` (name,image,mask,intacto,menor,mayor,destruido),
así ``train_damage_v3.py --extra-manifest`` lo consume sin cambios.

Uso:
    python prepare_rescuenet.py --split val --limit 20        # prueba
    python prepare_rescuenet.py --split train
    python prepare_rescuenet.py --split train --max-per-image 8
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cansat                                                       # noqa: F401
import cv2
import numpy as np

ROOT = Path("dataset/rescuenet")
OUT = Path("dataset/rescuenet_tiles")

# RescueNet (1..10) → clases del modelo (0/1/2)
REMAP = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}


def posiciones(dim: int, tile: int, stride: int) -> list[int]:
    """Orígenes de tile cubriendo el borde (el último tile pegado al final)."""
    if dim <= tile:
        return [0]
    pos = list(range(0, dim - tile + 1, stride))
    if pos[-1] != dim - tile:
        pos.append(dim - tile)
    return pos


def remap(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    for src, dst in REMAP.items():
        out[mask == src] = dst
    return out


def procesar_split(split: str, args) -> dict:
    img_dir = ROOT / f"{split}-org-img"
    lab_dir = ROOT / f"{split}-label-img"
    imgs = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        print(f"[ERROR] No hay imágenes en {img_dir}")
        return {}

    out_img = OUT / split / "images"
    out_msk = OUT / split / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    n_tiles = n_kept = 0
    for k, ip in enumerate(imgs, 1):
        lab = lab_dir / f"{ip.stem}_lab.png"
        img = cv2.imread(str(ip))
        msk = cv2.imread(str(lab), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print(f"  [WARN] no se pudo leer {ip.name}")
            continue
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        rm = remap(msk)

        candidatos = []
        for y0 in posiciones(img.shape[0], args.tile, args.stride):
            for x0 in posiciones(img.shape[1], args.tile, args.stride):
                t = rm[y0:y0 + args.tile, x0:x0 + args.tile]
                n_tiles += 1
                frac_b = float(((t >= 1).sum()) / t.size)
                if frac_b >= args.min_building:
                    candidatos.append((y0, x0, t, frac_b))
        if len(candidatos) > args.max_per_image:
            # prioriza los tiles con más construcción (señal de daño)
            candidatos.sort(key=lambda c: -c[3])
            candidatos = candidatos[: args.max_per_image]
            rng.shuffle(candidatos)

        for j, (y0, x0, t, _fb) in enumerate(candidatos):
            name = f"rescuenet_{split}_{ip.stem}_{y0}_{x0}"
            ti = int((t == 1).sum())
            tm = int((t == 2).sum())          # el remapeo colapsa 3/4/5 → 2
            if ti + tm == 0:
                continue
            cv2.imwrite(str(out_img / f"{name}.jpg"),
                        img[y0:y0 + args.tile, x0:x0 + args.tile],
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
            cv2.imwrite(str(out_msk / f"{name}.png"), t)
            rows.append({
                "name": name,
                "image": str(out_img / f"{name}.jpg"),
                "mask": str(out_msk / f"{name}.png"),
                "intacto": ti,
                "menor": 0,
                "mayor": 0,
                "destruido": tm,
            })
            n_kept += 1
        if k % 100 == 0 or k == len(imgs):
            print(f"  {k}/{len(imgs)} imágenes · tiles {n_kept}/{n_tiles} "
                  f"({100 * n_kept / max(1, n_tiles):.1f}%)", flush=True)

    man = OUT / f"manifest_{split}.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "image", "mask", "intacto",
                                          "menor", "mayor", "destruido"])
        w.writeheader()
        w.writerows(rows)
    stats = {
        "split": split,
        "imagenes": len(imgs),
        "tiles_generados": n_tiles,
        "tiles_guardados": n_kept,
        "min_building": args.min_building,
        "tile": args.tile,
        "manifest": str(man),
    }
    print(f"[OK] {split}: {n_kept} tiles en {man}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="RescueNet → tiles 0/1/2 + manifest")
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=640)
    ap.add_argument("--min-building", type=float, default=0.02,
                    help="fracción mínima de píxeles de edificio para guardar el tile")
    ap.add_argument("--max-per-image", type=int, default=8,
                    help="tope de tiles por imagen (prioriza los de más edificación)")
    ap.add_argument("--jpeg", type=int, default=85)
    ap.add_argument("--limit", type=int, default=0, help="solo N imágenes (prueba)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats = procesar_split(args.split, args)
    if stats:
        sp = OUT / f"stats_{args.split}.json"
        sp.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
        print(f"→ {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
