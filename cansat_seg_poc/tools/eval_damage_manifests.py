"""
Evalúa un checkpoint de daño (3 clases) sobre manifests con formato xBD.

Sirve para comparar manzanas con manzanas: el modelo viejo (xBD) y el nuevo
(xBD + RescueNet) medidos sobre EL MISMO manifest, con la métrica de la misión
(IoU de dañado two-stage: sólo sobre edificios).

Uso:
    python tools/eval_damage_manifests.py --checkpoint outputs/best_damage_v3.pth \
        --manifest dataset/rescuenet_tiles/manifest_val.csv
    python tools/eval_damage_manifests.py --checkpoint outputs/best_damage_v3_uav.pth \
        --manifest dataset/xbd_masks/manifest.csv --splits val --holdout-frac 0.2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cansat                                                       # noqa: E402,F401
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

from cansat.checkpoints import load_into                            # noqa: E402
from cansat.xbd import split_por_desastre                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Evalúa daño en un manifest")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", choices=("all", "val"), default="all",
                    help="all = todas las filas; val = holdout por desastre")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if args.splits == "val":
        _tr, rows, grupos = split_por_desastre(rows, args.holdout_frac, args.seed)
        print(f"Val por desastre: {grupos}")

    from train import DeepLabV3PlusMobileNetV2
    from train_damage_v3 import XBDv3, conf_matrix

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                       else "cuda")
    model = DeepLabV3PlusMobileNetV2(3)
    load_into(model, args.checkpoint, strict=False, min_loaded_frac=0.5)
    model = model.to(dev).eval()
    print(f"Checkpoint: {args.checkpoint} · device {dev} · {len(rows)} filas")

    dl = DataLoader(XBDv3(rows, args.size), batch_size=args.batch,
                    num_workers=0)
    cm = np.zeros((3, 3), dtype=np.int64)
    inter_d = union_d = 0
    with torch.no_grad():
        for x, y in dl:
            pred = model(x.to(dev)).argmax(1).cpu().numpy()
            yt = y.numpy()
            cm += conf_matrix(pred, yt)
            pd, gd, gb = pred == 2, yt == 2, yt >= 1
            inter_d += int((pd & gd).sum())
            union_d += int(((pd & gb) | gd).sum())

    iou_d2 = inter_d / union_d if union_d else 0.0
    ious = []
    for i in range(3):
        inter = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - inter
        ious.append(inter / union if union else 0.0)
    print(f"  other={ious[0]:.3f} intacto={ious[1]:.3f} dañado={ious[2]:.3f}")
    print(f"  DAÑADO* two-stage (sobre edificios) = {iou_d2:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
