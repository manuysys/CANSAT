"""
Sen1Floods11 → formato del especialista de flood (3 clases).

Lee el subset etiquetado a mano (446 chips Sentinel-2 L1C de 13 bandas +
LabelHand) del tarball de HuggingFace (blumenstiel/Sen1Floods11) y arma un
dataset compatible con `train_flood_specialist.py`:

  · RGB con B4/B3/B2 (rojo/verde/azul) escalado por percentiles 2-98 → PNG 8-bit.
  · Etiquetas remapeadas a nuestras 3 clases: 0=other, 1=flood, 2=agua-normal.
    ⚠ Sen1Floods11 NO distingue inundación de agua permanente en sus labels
    (QC: -1 no-data / 0 no-agua / 1 agua). Como los chips son de EVENTOS de
    inundación, el agua se mapea a `flood` (1) y el no-data a `other` (0);
    la clase `agua-normal` (2) no aparece en este dataset y se declara.
  · Splits OFICIALES (flood_train_data.txt / flood_valid_data.txt).

Salida: dataset/sen1floods11_remapped/{train,val}/{images,masks}/<chip>.png
        + summary.json con conteos y frecuencias de clase.

Uso:
    python prepare_sen1floods11.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

RAW = Path("dataset/sen1floods11_raw/sen1floods11_v1.1")
DST = Path("dataset/sen1floods11_remapped")
# Bandas del S2 L1C (0-based): B4 rojo=3, B3 verde=2, B2 azul=1.
BANDAS_RGB = (3, 2, 1)


def rgb_desde_s2(tif: np.ndarray) -> np.ndarray:
    """13 bandas uint16 (TOA x10000) → RGB uint8 con estiramiento 2-98 %."""
    rgb = tif[list(BANDAS_RGB)].astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        banda = rgb[c]
        lo, hi = np.percentile(banda, (2, 98))
        out[c] = np.clip((banda - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def remapear_label(lab: np.ndarray) -> np.ndarray:
    """QC (-1/0/1) → 3 clases: agua del evento → flood(1); resto → other(0)."""
    return (lab == 1).astype(np.uint8)


def main() -> int:
    if not RAW.is_dir():
        print(f"[ERROR] falta {RAW} (extraé el tarball de Sen1Floods11)")
        return 1

    resumen: dict = {"chips": {}, "frecuencias": {}}
    for split, txt in (("train", "flood_train_data.txt"),
                       ("val", "flood_valid_data.txt")):
        ids = [ln.strip() for ln in
               (RAW / "splits" / txt).read_text().splitlines() if ln.strip()]
        img_dir = DST / split / "images"
        msk_dir = DST / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        hist = np.zeros(3, dtype=np.int64)
        n = 0
        for chip in ids:
            s2 = RAW / "data" / "S2L1CHand" / f"{chip}_S2Hand.tif"
            lb = RAW / "data" / "LabelHand" / f"{chip}_LabelHand.tif"
            if not (s2.is_file() and lb.is_file()):
                continue
            rgb = rgb_desde_s2(tifffile.imread(s2))
            msk = remapear_label(tifffile.imread(lb))
            cv2.imwrite(str(img_dir / f"{chip}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(msk_dir / f"{chip}.png"), msk)
            hist += np.bincount(msk.ravel(), minlength=3)[:3]
            n += 1
        freq = (hist / max(1, hist.sum())).round(4).tolist()
        resumen["chips"][split] = n
        resumen["frecuencias"][split] = freq
        print(f"  {split}: {n} chips · frecuencias [other, flood, agua]={freq}")

    (DST / "summary.json").write_text(json.dumps(resumen, indent=2),
                                      encoding="utf-8")
    print(f"[OK] {DST} listo ({resumen['chips']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
