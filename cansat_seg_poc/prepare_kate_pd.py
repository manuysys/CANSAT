"""
CanSat La Base — F2 v3: prepara KATE-PD (terremoto Türkiye 2023) para daño.

KATE-PD (HF `cscrs/kate-pd`, IGARSS 2025) trae tiles de 512² post-terremoto
con polígonos YOLO de edificios DAÑADOS y una máscara binaria (0/255).

Se convierte a nuestro esquema de 3 clases para el modelo de daño:
    0 = other (fondo y edificios NO dañados — KATE-PD no los etiqueta)
    2 = dañado (los polígonos del dataset)
y se escribe un manifest con el mismo formato que xBD/RescueNet.

⚠ KATE-PD no tiene clase "intacto": se usa como **test cross-event de sismo**
  (el modelo nunca vio Türkiye 2023) y, opcionalmente, como datos de la clase
  dañado para el entrenamiento.

Uso:
    python prepare_kate_pd.py
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cansat                                                       # noqa: E402,F401
import numpy as np                                                  # noqa: E402
import pyarrow.parquet as pq                                        # noqa: E402
from PIL import Image                                               # noqa: E402

SRC = ROOT / "dataset/kate_pd/data"
DST = ROOT / "dataset/kate_pd_tiles"
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def main() -> int:
    total = 0
    for split, out_name in SPLITS.items():
        p = SRC / f"{split}-00000-of-00001.parquet"
        if not p.is_file():
            print(f"[WARN] falta {p}")
            continue
        out_img = DST / out_name / "images"
        out_msk = DST / out_name / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        pf = pq.ParquetFile(p)
        rows: list[dict] = []
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg)
            for i in range(t.num_rows):
                name = t["name"][i].as_py()
                img = Image.open(io.BytesIO(t["image"][i].as_py()["bytes"])).convert("RGB")
                msk = Image.open(io.BytesIO(t["mask"][i].as_py()["bytes"])).convert("L")
                m = (np.array(msk) > 127).astype(np.uint8) * 2   # 0/255 → 0/2
                stem = Path(name).stem
                img.save(out_img / f"{stem}.jpg", quality=92)
                Image.fromarray(m).save(out_msk / f"{stem}.png")
                n_dan = int((m == 2).sum())
                rows.append({
                    "name": stem,
                    "image": str(out_img / f"{stem}.jpg"),
                    "mask": str(out_msk / f"{stem}.png"),
                    "intacto": 0, "menor": 0, "mayor": 0, "destruido": n_dan,
                })
        man = DST / f"manifest_{out_name}.csv"
        with man.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "image", "mask",
                                              "intacto", "menor", "mayor",
                                              "destruido"])
            w.writeheader()
            w.writerows(rows)
        n_px = sum(r["destruido"] for r in rows)
        print(f"[OK] {out_name}: {len(rows)} tiles · dañado {n_px} px → {man}")
        total += len(rows)
    print(f"[OK] total {total} tiles en {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
