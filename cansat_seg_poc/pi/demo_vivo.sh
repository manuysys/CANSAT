#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  demo_vivo.sh — captura EN VIVO para la presentación (se corre EN la Pi)
# ═══════════════════════════════════════════════════════════════════════════
#  Cadena: Heltec (UART GPIO15, sensores reales) → listener → pipeline con la
#  AI Camera (tiny@224 + detección on-sensor IMX500) → misión lista para que
#  la PC haga el post-vuelo completo (tools/demo_pi.ps1).
#
#  Uso:
#      bash pi/demo_vivo.sh            # 12 frames (~45 s)
#      bash pi/demo_vivo.sh 20         # 20 frames
#
#  Requisitos: Heltec emitiendo por GPIO15 (si no hay paquetes, avisa y sigue
#  con atmósfera simulada: el pipeline no se cae) y la AI Camera conectada.
set -euo pipefail
cd "$(dirname "$0")/.."

FRAMES="${1:-12}"
OUT="${2:-$HOME/vuelos/demo_$(date +%Y%m%d_%H%M%S)}"
STATE="$OUT/uart_state.json"
mkdir -p "$OUT"

echo "════════ DEMO EN VIVO ════════"
echo "  frames : $FRAMES"
echo "  salida : $OUT"
echo "  UART   : /dev/serial0 (Heltec por GPIO15)"

# 1) Listener UART en segundo plano (deja uart_state.json fresco para el pipeline).
venv/bin/python uart_listener.py --port /dev/serial0 \
    --out "$OUT/uart_log.jsonl" --state "$STATE" --overwrite \
    > "$OUT/uart_listener.log" 2>&1 &
LISTENER=$!
sleep 3
if [ -s "$STATE" ]; then
    echo "  UART   : OK, llegaron paquetes de la Heltec"
else
    echo "  [WARN] UART: sin paquetes (¿Heltec desconectada?); se usa atmósfera simulada"
fi

# 2) Captura + IA de vuelo (tiny + detección on-sensor). Sin daño/flood/fuego:
#    esos van en el post-vuelo de la PC (que es rápida con onnxruntime).
venv/bin/python mission_pipeline.py --camera --frames "$FRAMES" --interval 0 \
    --no-damage --overwrite \
    --onnx outputs/cansat_seg_terrain_tiny_224.onnx --img-size 224 \
    --det-backend imx500 --shutter 8000 --gain 16 \
    --uart-state "$STATE" \
    --out-dir "$OUT/mission" | tail -8

# 3) Cierre del listener.
kill "$LISTENER" 2>/dev/null || true
wait "$LISTENER" 2>/dev/null || true

echo
echo "════════ MISIÓN LISTA ════════"
echo "MISION=$OUT/mission"
ls "$OUT/mission"
echo
echo "  personas detectadas (JSONL):"
grep -o '"tipo": "persona"' "$OUT/mission/telemetry.jsonl" 2>/dev/null | wc -l || true
