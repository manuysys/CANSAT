#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  CanSat LB135 — Instalador del entorno de vuelo (se corre EN la Raspberry Pi)
# ═══════════════════════════════════════════════════════════════════════════
#  Detecta la arquitectura:
#   · armv6l  (Pi Zero W v1)  → cv2 del sistema + venv --system-site-packages,
#                               sin onnxruntime ni ultralytics (no existen).
#   · aarch64 (Zero 2 W 64-bit / Pi 4 / Pi 5) → pip install -r requirements.
#
#  Uso (en la Pi, dentro de la carpeta del proyecto):
#      bash pi/instalar_en_pi.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

MACHINE=$(uname -m)
echo "════════════════════════════════════════════════════════════"
echo "  Arquitectura detectada: $MACHINE"
echo "════════════════════════════════════════════════════════════"

if [ ! -f mission_pipeline.py ]; then
    echo "[ERROR] Ejecutá este script desde la raíz del proyecto (donde está"
    echo "        mission_pipeline.py).  cd ~/cansat_seg_poc && bash pi/instalar_en_pi.sh"
    exit 1
fi

echo "── 1/4  Paquetes del sistema ──"
sudo apt update
if [ "$MACHINE" = "armv6l" ]; then
    sudo apt install -y python3-venv python3-picamera2 python3-opencv
else
    sudo apt install -y python3-venv python3-picamera2
fi

echo "── 2/4  Entorno virtual ──"
if [ "$MACHINE" = "armv6l" ]; then
    # --system-site-packages: el venv necesita ver picamera2 y cv2 de apt.
    python3 -m venv --system-site-packages venv
else
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip --quiet

echo "── 3/4  Dependencias de Python ──"
if [ "$MACHINE" = "armv6l" ]; then
    echo "    ARMv6: sin onnxruntime ni ultralytics (no hay wheels)."
    echo "    El pipeline usa el backend cv2.dnn de cansat/onnxio.py."
    pip install pyserial "PyYAML>=6.0"
else
    pip install -r requirements-flight.txt
fi

echo "── 4/4  Verificación ──"
python - <<'PY'
import sys
import cv2
import numpy
try:
    import serial
    tiene_serial = True
except ImportError:
    tiene_serial = False
try:
    import onnxruntime  # noqa: F401
    tiene_ort = True
except ImportError:
    tiene_ort = False
print(f"    python     {sys.version.split()[0]} ({sys.platform}, {__import__('platform').machine()})")
print(f"    cv2        {cv2.__version__}")
print(f"    numpy      {numpy.__version__}")
print(f"    pyserial   {'sí' if tiene_serial else 'NO'}")
print(f"    onnxruntime {'sí' if tiene_ort else 'no (se usará cv2.dnn)'}")
falta_picam = True
try:
    import picamera2  # noqa: F401
    falta_picam = False
except ImportError:
    pass
print(f"    picamera2  {'sí' if not falta_picam else 'NO (--camera no funcionará)'}")
PY

echo
echo "✔ Listo. Próximo paso (medir velocidad y probar):"
echo "    source venv/bin/activate"
echo "    time python mission_pipeline.py --folder <carpeta-de-tiles> --frames 3 \\"
echo "         --interval 0 --no-detect --no-damage --overwrite"
echo
echo "  ⚠ Los modelos deben ser AUTOCONTENIDOS en la Zero W v1: generarlos en la"
echo "    PC con  python tools/onnx_inline.py ...  (ver pi/guia_pi.md, paso 5)."
