"""
DEPRECADO — usar el exportador único:

    python export_onnx.py --checkpoint outputs/best_terrain_segformer_b5.pth \\
                          --output outputs/cansat_seg_terrain_segformer_b5_512.onnx \\
                          --arch segformer --num-classes 5 --img-size 512

Había SEIS scripts de exportación que eran el mismo torch.onnx.export con
distinta ruta, y no eran equivalentes: sólo dos pasaban dynamo=False (que
docs/reporte_pruebas.md declara requisito del conversor Sony IMX500) y cada
uno servía para un único formato de checkpoint.

Este script en particular, además:

  · instanciaba el modelo desde HuggingFace en runtime (requiere red) y
    con `ignore_mismatched_sizes=True`, que era innecesario porque
    inmediatamente después se pisaba todo con `load_state_dict`.


Se mantiene como shim para no romper la documentación existente.
"""
from __future__ import annotations

import sys

CKPT = "outputs/best_terrain_segformer_b5.pth"
OUT = "outputs/cansat_seg_terrain_segformer_b5_512.onnx"
ARCH = "segformer"
NUM_CLASSES = 5
IMG_SIZE = 512


if __name__ == "__main__":
    sys.stderr.write(f"[DEPRECADO] export_b5_512.py → python export_onnx.py --checkpoint {CKPT} "
                     f"--output {OUT} --arch {ARCH}\n")
    from export_onnx import export
    raise SystemExit(export(CKPT, OUT, ARCH, NUM_CLASSES, IMG_SIZE))
