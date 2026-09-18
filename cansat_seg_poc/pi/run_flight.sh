#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Vuelo autónomo: captura + IA + telemetría, logueado en la SD.
# ═══════════════════════════════════════════════════════════════════════════
#  ⚠ Pi Zero W v1 (ARMv6):
#    · --no-detect se reemplaza por --det-backend imx500 si la cámara es la
#      AI Camera (el NPU corre la detección sin cargar la CPU).
#    · --no-damage es lo recomendado hasta medir los modelos por separado.
#      (También apaga el flood y el fuego/humo: corren en el bloque de daño.)
#    · --onnx ..._224.onnx + --img-size 224: el terreno a 224 px cuesta la
#      mitad que a 320 (mIoU 0.4996 vs 0.5219). Si sobra tiempo de frame,
#      volver al de 320 (--onnx outputs/cansat_seg_terrain_v2.onnx --img-size 320).
#    · --frames 1000 equivale a "grabar hasta que se corte la energía"; la
#      telemetría se flushea en cada frame.
#    · --pop-density: densidad del predio para la estimación de pérdidas
#      humanas (DPD). AJUSTAR con el dato real (INDEC/WorldPop).
#
#  Arranque automático:  crontab -e  →  @reboot /home/pi/cansat_seg_poc/pi/run_flight.sh
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

cd /home/pi/cansat_seg_poc || { echo "[ERROR] falta /home/pi/cansat_seg_poc"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate || { echo "[ERROR] falta el venv (correr pi/instalar_en_pi.sh)"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT="/home/pi/vuelos/$TS"
mkdir -p "$OUT"

echo "[$(date -Is)] inicio de vuelo → $OUT" >> "$OUT/vuelo.log"

if python mission_pipeline.py --camera --frames 1000 --interval 0 \
        --enhance --no-damage --det-backend imx500 --pop-density 1500 \
        --onnx outputs/cansat_seg_terrain_v2_224.onnx --img-size 224 \
        --uart-state outputs/uart_state.json \
        --out-dir "$OUT" >> "$OUT/vuelo.log" 2>&1; then
    echo "[$(date -Is)] vuelo terminado OK" >> "$OUT/vuelo.log"
else
    echo "[$(date -Is)] [ERROR] el pipeline terminó con exit=$?" >> "$OUT/vuelo.log"
fi
