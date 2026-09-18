"""
CanSat La Base — F3: prepara el dataset de fuego/humo (LibreYOLO/fire-smoke-seg).

Fuente: HuggingFace `LibreYOLO/fire-smoke-seg` (derivado de FLAME, CC BY 4.0),
formato YOLO-seg: `clase x1 y1 x2 y2 ...` normalizado, imágenes 256×256.

Salida: `dataset/fire_smoke/{split}/{images,masks}` + `manifest_{split}.csv`
con el formato del resto de los manifests (name,image,mask + conteos por clase).

Clases del modelo (3): 0 = other, 1 = fuego, 2 = humo.

Uso:
    python prepare_fire_smoke.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cansat                                                       # noqa: F401
import cv2
import numpy as np

SRC = Path("dataset/fire_smoke_seg")
DST = Path("dataset/fire_smoke")
SPLITS = ("train", "valid", "test")


def poligonos_a_mascara(txt: Path, w: int, h: int) -> np.ndarray:
    """YOLO-seg (clase + puntos normalizados) → máscara 0/1/2."""
    msk = np.zeros((h, w), dtype=np.uint8)
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:          # clase + al menos 3 puntos
            continue
        cls = int(parts[0]) + 1     # 0=fuego→1, 1=humo→2
        pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        cv2.fillPoly(msk, [np.round(pts).astype(np.int32)], int(cls))
    return msk


def main() -> int:
    total = 0
    for split in SPLITS:
        src_img = SRC / split / "images"
        src_lab = SRC / split / "labels"
        if not src_img.is_dir():
            print(f"[WARN] falta {src_img}")
            continue
        out_img = DST / split / "images"
        out_msk = DST / split / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        rows = []
        for ip in sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png")):
            lp = src_lab / f"{ip.stem}.txt"
            if not lp.is_file():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            h, w = img.shape[:2]
            msk = poligonos_a_mascara(lp, w, h)
            cv2.imwrite(str(out_img / f"{ip.stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out_msk / f"{ip.stem}.png"), msk)
            rows.append({
                "name": ip.stem,
                "image": str(out_img / f"{ip.stem}.jpg"),
                "mask": str(out_msk / f"{ip.stem}.png"),
                "fuego": int((msk == 1).sum()),
                "humo": int((msk == 2).sum()),
            })
        man = DST / f"manifest_{split}.csv"
        with man.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["name", "image", "mask",
                                               "fuego", "humo"])
            wr.writeheader()
            wr.writerows(rows)
        n_f = sum(r["fuego"] for r in rows)
        n_s = sum(r["humo"] for r in rows)
        print(f"[OK] {split}: {len(rows)} pares · fuego {n_f} px · humo {n_s} px "
              f"→ {man}")
        total += len(rows)
    print(f"[OK] total {total} pares en {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
