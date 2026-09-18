"""
DEPRECADO — reemplazado por ``evaluate.py``.


Su docstring prometía "Compara mIoU en Val: checkpoint FP32 vs
inferencia cuantizada", pero el código sólo evaluaba el FP32: la comparación
no existía. Tenía además un import sys muerto y la matriz de confusión armada
con doble loop de Python.


Se mantiene como shim para no romper referencias de documentación. Toda la
funcionalidad está en `evaluate.py`, que escribe los resultados a
`outputs/metrics/val_<timestamp>.json` con el hash del modelo, la semilla y el
tamaño real de la muestra.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "[DEPRECADO] eval_onnx_gap.py → usar evaluate.py\n"
        "  Ejemplo:  python evaluate.py --onnx outputs/cansat_seg_terrain_v2.onnx\n"
        "  Compara:  python evaluate.py --onnx <fp32> --compare <int8>\n")
    from evaluate import main
    raise SystemExit(main(sys.argv[1:]))
