#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_flight.sh — vuelo autónomo (se corre EN la Pi)
# ═══════════════════════════════════════════════════════════════════════════
#  Cadena: Heltec (UART GPIO15, sensores reales) → listener → pipeline con la
#  AI Camera (tiny@224 + detección YOLO11n on-sensor) → telemetría en la SD.
#  El daño/flood/fuego/severidad corren POST-VUELO en la PC.
#
#  Uso manual:     FRAMES=20 bash pi/run_flight.sh
#  Arranque auto:  sudo systemctl enable cansat-vuelo   (ver guia §7)
#                  (o crontab -e → @reboot /home/pi/cansat_seg_poc/pi/run_flight.sh)
#
#  Variables: FRAMES (default 1000 = hasta cortar energía), OUT_BASE, DET_MODEL.
set -uo pipefail
cd /home/pi/cansat_seg_poc || { echo "[ERROR] falta /home/pi/cansat_seg_poc"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate || { echo "[ERROR] falta el venv (correr pi/instalar_en_pi.sh)"; exit 1; }

FRAMES="${FRAMES:-1000}"
OUT_BASE="${OUT_BASE:-/home/pi/vuelos}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$OUT_BASE/$TS"
STATE="$OUT/uart_state.json"
mkdir -p "$OUT"

# Detector del NPU: YOLO11n (mAP 0.374) si está; si no, el SSD del apt (0.218).
DET_MODEL="${DET_MODEL:-/home/pi/cansat_seg_poc/models_rpk/yolo11n_pp.rpk}"
[ -f "$DET_MODEL" ] || DET_MODEL="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

echo "[$(date -Is)] inicio de vuelo → $OUT ($FRAMES frames)" | tee -a "$OUT/vuelo.log"

# Listener UART (si la Heltec no emite, el pipeline sigue con atmósfera
# simulada: no se cae).
python uart_listener.py --port /dev/serial0 \
    --out "$OUT/uart_log.jsonl" --state "$STATE" --overwrite \
    > "$OUT/uart_listener.log" 2>&1 &
LISTENER=$!
sleep 3
if [ -s "$STATE" ]; then
    echo "[$(date -Is)] UART OK: llegaron paquetes de la Heltec" >> "$OUT/vuelo.log"
else
    echo "[$(date -Is)] [WARN] UART sin paquetes: atmósfera simulada" >> "$OUT/vuelo.log"
fi

python mission_pipeline.py --camera --frames "$FRAMES" --interval 0 \
    --no-damage --overwrite \
    --onnx outputs/cansat_seg_terrain_tiny_224.onnx --img-size 224 \
    --det-backend imx500 --imx500-model "$DET_MODEL" --shutter 8000 --gain 16 \
    --uart-state "$STATE" \
    --pop-density 1500 \
    --out-dir "$OUT/mission" >> "$OUT/vuelo.log" 2>&1
RC=$?
kill "$LISTENER" 2>/dev/null || true
wait "$LISTENER" 2>/dev/null || true

if [ "$RC" -eq 0 ]; then
    echo "[$(date -Is)] vuelo terminado OK ($FRAMES frames)" >> "$OUT/vuelo.log"
else
    echo "[$(date -Is)] [ERROR] el pipeline terminó con exit=$RC" >> "$OUT/vuelo.log"
fi
exit "$RC"
