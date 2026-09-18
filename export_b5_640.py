"""
DEPRECADO — usar el exportador único:

    python export_onnx.py --checkpoint outputs/best_terrain_segformer_b5.pth \\
                          --output outputs/cansat_seg_terrain_segformer_b5_640.onnx \\
                          --arch segformer --num-classes 5 --img-size 640

Había SEIS scripts de exportación que eran el mismo torch.onnx.export con
distinta ruta, y no eran equivalentes: sólo dos pasaban dynamo=False (que
docs/reporte_pruebas.md declara requisito del conversor Sony IMX500) y cada
uno servía para un único formato de checkpoint.

Este script en particular, además:
  · cargaba `outputs/best_terrain_b5.pth`, que **no existe** en el repo
    (el real es `best_terrain_segformer_b5.pth`);
  · instanciaba `SegformerTerrain`, una clase que **no existe** en
    `train_segformer_b5.py` (ahí sólo hay `Wrap`).
    → No corría. Y como `post_flight.py` apuntaba al `.onnx` que este script
      debía generar, toda la segunda pasada de alta calidad se salteaba en
      silencio.



Se mantiene como shim para no romper la documentación existente.
"""
from __future__ import annotations

import sys

CKPT = "outputs/best_terrain_segformer_b5.pth"
OUT = "outputs/cansat_seg_terrain_segformer_b5_640.onnx"
ARCH = "segformer"
NUM_CLASSES = 5
IMG_SIZE = 640


if __name__ == "__main__":
    sys.stderr.write(f"[DEPRECADO] export_b5_640.py → python export_onnx.py --checkpoint {CKPT} "
                     f"--output {OUT} --arch {ARCH}\n")
    from export_onnx import export
    raise SystemExit(export(CKPT, OUT, ARCH, NUM_CLASSES, IMG_SIZE))
