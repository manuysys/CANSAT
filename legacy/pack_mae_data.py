"""
CanSat La Base — Arma mae_pack/images con ~8k recortes 256px sin labels.
Uso: python pack_mae_data.py
Después copiá adentro de mae_pack/: mae_pretrain.py, mae_finetune_terrain.py
y LEEME_MAE.txt → zippeá mae_pack y entregalo.
"""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/pack_mae_data.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import random
from pathlib import Path

import cv2

OUT = Path("mae_pack/images")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("dataset/loveda_raw", 4000),
    ("dataset/floodnet", 2500),
    ("dataset/xbd", 1000),
    ("datasets/VisDrone/images/train", 1000),
]

n = 0
for root, cap in SOURCES:
    r = Path(root)
    if not r.exists():
        print(f"[skip] {root}")
        continue
    files = [p for p in r.rglob("*")
             if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
    random.seed(7)
    random.shuffle(files)
    k = 0
    for p in files[:cap]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(str(OUT / f"mae_{n:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        n += 1
        k += 1
    print(f"[OK] {root}: {k}")
print(f"[LISTO] {n} imágenes en {OUT}")
