#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Conversión al NPU del IMX500 — CanSat LB135
# ═══════════════════════════════════════════════════════════════════════════
#  ⚠ ESTADO: NO VERIFICADO. Este script es una GUÍA; el flujo oficial de Sony
#    (Edge-MDT) se corre en una PC Linux con ≥4 GB de RAM y Python 3.11, NO en
#    la Pi. Antes de usarlo:
#
#      1. Instalar Edge-MDT en la PC Linux (ver docs/DATASETS-Y-TECNICAS.md §3.3):
#           pip install edge-mdt        # trae MCT + imxconv-pt + packager
#      2. Los modelos tienen que ser AUTOCONTENIDOS y opset 17:
#           python audit_imx500.py --all
#         (los de vuelo ya pasan: terrain_v2, damage3, damage_v3, flood)
#      3. Flujo real (3 pasos, en la PC Linux):
#           python -m model_compression_toolkit ...   # MCT: cuantiza+comprime
#           imxconv-pt -i <model_mct.onnx> -o out/    # compila a IMX500
#           imx500-package -i out/packerOut.zip -o rpk/   # empaqueta .rpk
#      4. Copiar el .rpk a la Pi y correr:
#           python -m cansat.imx500 --model /home/pi/rpk/network.rpk
#
#  La sintaxis exacta depende de la versión de Edge-MDT: si `imxconv-pt --help`
#  no coincide, pegar la salida en el issue del repo y ajustar.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

cd /home/pi/cansat_seg_poc || { echo "[ERROR] falta /home/pi/cansat_seg_poc"; exit 1; }

echo "════════════════════════════════════════════════════════════════"
echo "  Conversión IMX500 — leer los avisos del encabezado del script"
echo "════════════════════════════════════════════════════════════════"
echo
echo "Modelos de vuelo candidatos (deben pasar audit_imx500.py):"
echo "  · outputs/cansat_seg_terrain_v2.onnx   (segmentación, 52.19 % mIoU)"
echo "  · outputs/cansat_seg_terrain_v2_224.onnx (alternativa 224 px, 49.96 %)"
echo "  · outputs/cansat_seg_terrain_tiny_224.onnx (tiny 4.3 MB, candidato NPU)"
echo "  · outputs/cansat_flood_specialist.onnx (flood, ya auditado)"
echo "  · outputs/cansat_damage3_mobilenetv2.onnx (daño principal, xBD)"
echo "  · outputs/cansat_damage_v3_bal.onnx (daño two-stage de vuelo, UAV)"
echo
echo "⚠ El .rpk se genera en la PC Linux con Edge-MDT, no acá."
echo "  Ver docs/DATASETS-Y-TECNICAS.md §3.3 y docs/SUSTITUCION-HW.md §4."
echo
echo "Para verificar los requisitos desde la Pi:"
echo "  python audit_imx500.py --all"
