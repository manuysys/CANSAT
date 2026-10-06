#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  preflight.sh — chequeo PREVIO AL VUELO (se corre EN la Pi, en el predio)
# ═══════════════════════════════════════════════════════════════════════════
#  Verifica todo lo que el vuelo necesita y resume PASS/WARN/FAIL:
#    · modelos (terreno tiny, YOLO11n, daño ep6, rpk propio)
#    · cámara + NPU (foto de prueba con el post-process del SSD)
#    · UART con la Heltec (paquetes válidos en 8 s)
#    · espacio en la microSD (la guarda del pipeline degrada a <150 MB)
#    · temperatura y throttling del SoC
#    · reloj (sin RTC el boot arranca con hora vieja)
#    · servicio systemd del vuelo (si está instalado)
#
#  Uso:  bash pi/preflight.sh
set -uo pipefail
cd /home/pi/cansat_seg_poc || exit 1

PASS=0; WARN=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "════════ PRE-VUELO ════════"

# ── 1. Modelos ─────────────────────────────────────────────────────────
[ -f outputs/cansat_seg_terrain_tiny_224.onnx ] \
    && ok "modelo de vuelo (tiny@224)" || fail "falta el modelo de vuelo"
[ -f models_rpk/yolo11n_pp.rpk ] \
    && ok "detector YOLO11n (.rpk)" || warn "sin yolo11n_pp.rpk (cae al SSD)"
[ -f outputs/cansat_damage_v3_bal_ep6.onnx ] \
    && ok "daño ep6 (post-vuelo)" || warn "sin el daño ep6 (post-vuelo)"
[ -f /home/pi/modelos/network_compacto.rpk ] \
    && ok "rpk propio de terreno" || warn "sin el rpk propio"

# ── 2. Cámara + NPU ────────────────────────────────────────────────────
if timeout 40 rpicam-still -n -t 500 \
        --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json \
        -o /tmp/preflight_cam.jpg >/dev/null 2>&1 && [ -s /tmp/preflight_cam.jpg ]; then
    ok "cámara IMX500 + NPU (foto de prueba)"
else
    fail "cámara/NPU no respondió (revisar flex CSI)"
fi

# ── 3. UART con la Heltec (8 s) ────────────────────────────────────────
if [ -c /dev/serial0 ]; then
    # 12 s de ventana: el listener tarda ~3 s en arrancar (imports + puerto).
    rm -f /tmp/preflight_uart.jsonl
    timeout 12 venv/bin/python uart_listener.py --port /dev/serial0 \
        --out /tmp/preflight_uart.jsonl --overwrite --quiet >/dev/null 2>&1 || true
    NPKT=$(wc -l < /tmp/preflight_uart.jsonl 2>/dev/null || echo 0)
    if [ "$NPKT" -ge 6 ]; then
        ok "UART: $NPKT paquetes en 8 s"
    elif [ "$NPKT" -ge 1 ]; then
        warn "UART: sólo $NPKT paquetes (¿Heltec con poca señal?)"
    else
        fail "UART: 0 paquetes (¿Heltec encendida? TX→GPIO15, GND común)"
    fi
else
    fail "no existe /dev/serial0 (revisar raspi-config)"
fi

# ── 4. Espacio en SD ───────────────────────────────────────────────────
LIBRE=$(df --output=avail -m /home/pi | tail -1 | tr -d ' ')
if [ "${LIBRE:-0}" -ge 2000 ]; then
    ok "SD: ${LIBRE} MB libres"
elif [ "${LIBRE:-0}" -ge 500 ]; then
    warn "SD: sólo ${LIBRE} MB libres (un vuelo de 30 min ~1 GB)"
else
    fail "SD: ${LIBRE} MB libres (vaciar antes de volar)"
fi

# ── 5. Temperatura y throttling ────────────────────────────────────────
TEMP=$(vcgencmd measure_temp | tr -dc '0-9.')
THR=$(vcgencmd get_throttled)
if [ "${THR#*=}" = "0x0" ]; then
    ok "throttling: 0x0 · temp ${TEMP} °C"
else
    warn "throttled=${THR#*=} (¿fuente débil o calor?) · temp ${TEMP} °C"
fi

# ── 6. Reloj ───────────────────────────────────────────────────────────
YEAR=$(date +%Y)
if [ "$YEAR" -ge 2026 ]; then
    ok "reloj: $(date '+%Y-%m-%d %H:%M')"
else
    fail "reloj en $YEAR (ajustar hora antes de volar: los timestamps lo usan)"
fi

# ── 7. systemd (informativo) ───────────────────────────────────────────
# ⚠ Sin `grep -q`: con `set -o pipefail`, grep -q cierra el pipe al primer
# match y systemctl muere con SIGPIPE (141) → el if daba falso siempre.
UNITS=$(systemctl list-unit-files 2>/dev/null || true)
if echo "$UNITS" | grep cansat-vuelo >/dev/null; then
    # is-enabled sale con 1 si está disabled: capturar sin `||` para no
    # duplicar la salida ("disabled" + "?").
    EST=$(systemctl is-enabled cansat-vuelo 2>/dev/null || true)
    ok "systemd: cansat-vuelo (${EST:-?})"
else
    warn "systemd: cansat-vuelo no instalado (arranque por crontab/manual)"
fi

echo "───────────────────────────"
echo "  PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "  → NO VOLAR todavía: resolver los FAIL"
    exit 1
fi
echo "  → LISTO para volar"
