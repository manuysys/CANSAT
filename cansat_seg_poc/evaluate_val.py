"""
DEPRECADO — reemplazado por ``evaluate.py``.

Evaluaba el ONNX sobre LoveDA Val, pero por default usaba sólo 100
imágenes de ~1669 e imprimía al final un mIoU de entrenamiento hardcodeado
(52.44%) que no salía de ninguna medición.



Se mantiene como shim para no romper referencias de documentación. Toda la
funcionalidad está en `evaluate.py`, que escribe los resultados a
`outputs/metrics/val_<timestamp>.json` con el hash del modelo, la semilla y el
tamaño real de la muestra.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "[DEPRECADO] evaluate_val.py → usar evaluate.py\n"
        "  Ejemplo:  python evaluate.py --onnx outputs/cansat_seg_terrain_v2.onnx\n"
        "  Compara:  python evaluate.py --onnx <fp32> --compare <int8>\n")
    from evaluate import main
    raise SystemExit(main(sys.argv[1:]))
