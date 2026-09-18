"""
DEPRECADO — reemplazado por ``evaluate.py``.



Prueba rápida de 50 frames INT8 en CPU, sin main() ni argparse (todo
a nivel módulo, así que se ejecutaba al importarlo), con 1669 hardcodeado en
la estimación de tiempo y un print por frame.

Se mantiene como shim para no romper referencias de documentación. Toda la
funcionalidad está en `evaluate.py`, que escribe los resultados a
`outputs/metrics/val_<timestamp>.json` con el hash del modelo, la semilla y el
tamaño real de la muestra.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "[DEPRECADO] eval_int8_cpu.py → usar evaluate.py\n"
        "  Ejemplo:  python evaluate.py --onnx outputs/cansat_seg_terrain_v2.onnx\n"
        "  Compara:  python evaluate.py --onnx <fp32> --compare <int8>\n")
    from evaluate import main
    raise SystemExit(main(sys.argv[1:]))
