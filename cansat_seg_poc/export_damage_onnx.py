"""
DEPRECADO — usar el exportador único:

    python export_onnx.py --checkpoint outputs/best_damage3.pth \\
                          --output outputs/cansat_damage3_mobilenetv2.onnx \\
                          --arch damage --num-classes 3 --img-size 320

Había SEIS scripts de exportación que eran el mismo torch.onnx.export con
distinta ruta, y no eran equivalentes: sólo dos pasaban dynamo=False (que
docs/reporte_pruebas.md declara requisito del conversor Sony IMX500) y cada
uno servía para un único formato de checkpoint.

Este script en particular, además:




Se mantiene como shim para no romper la documentación existente.
"""
from __future__ import annotations

import sys

CKPT = "outputs/best_damage3.pth"
OUT = "outputs/cansat_damage3_mobilenetv2.onnx"
ARCH = "damage"
NUM_CLASSES = 3
IMG_SIZE = 320


if __name__ == "__main__":
    sys.stderr.write(f"[DEPRECADO] export_damage_onnx.py → python export_onnx.py --checkpoint {CKPT} "
                     f"--output {OUT} --arch {ARCH}\n")
    from export_onnx import export
    raise SystemExit(export(CKPT, OUT, ARCH, NUM_CLASSES, IMG_SIZE))
