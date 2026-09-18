"""
DEPRECADO — usar el exportador único:

    python export_onnx.py --checkpoint outputs/best_terrain_v2.pth \\
                          --output outputs/cansat_seg_terrain_v2.onnx \\
                          --arch deeplabv3plus --num-classes 5 --img-size 320

Había SEIS scripts de exportación que eran el mismo torch.onnx.export con
distinta ruta, y no eran equivalentes: sólo dos pasaban dynamo=False (que
docs/reporte_pruebas.md declara requisito del conversor Sony IMX500) y cada
uno servía para un único formato de checkpoint.

Este script en particular, además:


  · parchaba una confusión de firma con `try: M(N) except TypeError: M()`,
    síntoma de que los dos formatos de checkpoint ya estaban causando problemas.

Se mantiene como shim para no romper la documentación existente.
"""
from __future__ import annotations

import sys

CKPT = "outputs/best_terrain_v2.pth"
OUT = "outputs/cansat_seg_terrain_v2.onnx"
ARCH = "deeplabv3plus"
NUM_CLASSES = 5
IMG_SIZE = 320


if __name__ == "__main__":
    sys.stderr.write(f"[DEPRECADO] export_v2.py → python export_onnx.py --checkpoint {CKPT} "
                     f"--output {OUT} --arch {ARCH}\n")
    from export_onnx import export
    raise SystemExit(export(CKPT, OUT, ARCH, NUM_CLASSES, IMG_SIZE))
