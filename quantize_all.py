"""
CanSat La Base — Cuantización dinámica INT8 de los modelos de vuelo.

⚠ EL DOCSTRING ANTERIOR PROMETÍA *"~2x más rápido en CPU de la Pi, ~1% de
  pérdida de precisión"*. Ninguno de los dos números estaba medido, y el primero
  es poco realista: ``quantize_dynamic`` cuantiza **sólo los pesos** (MatMul /
  Conv), las activaciones siguen en FP32. En una CNN convolucional el costo está
  dominado por las convoluciones con activaciones, así que la ganancia típica es
  mucho menor que 2× — a veces nula. Para ganar velocidad de verdad hace falta
  cuantización **estática** (``quantize_onnx.py``) o QAT.

  La pérdida tampoco es "~1%": el proyecto midió 87.6% de acuerdo en una
  validación que además era inválida (ver ``validate_int8_mission.py``). Medir
  antes de afirmar:

      python evaluate.py --onnx <fp32> --compare <int8>
      python validate_int8_mission.py

⚠ Además este script escribe ``*_int8.onnx`` mientras ``quantize_onnx.py``
  escribía ``*_int8_calib.onnx``, y ``mission_pipeline.py`` buscaba el primero.
  Dos pipelines de cuantización con salidas distintas y ninguna documentación de
  cuál vuela. Ahora ambos escriben ``*_int8.onnx`` y ``MODELS.yaml`` registra
  cuál se generó cómo.

Uso:
    python quantize_all.py
    python quantize_all.py --only outputs/cansat_seg_terrain_v2.onnx
"""
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

MODELS = [
    "outputs/cansat_seg_terrain_v2.onnx",
    "outputs/cansat_damage3_mobilenetv2.onnx",
    "outputs/cansat_damage_v3.onnx",
    "outputs/cansat_flood_specialist.onnx",
    "models/yolov8n_visdrone.onnx",
]

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Cuantización dinámica INT8")
    ap.add_argument("--only", default=None, help="cuantizar un solo modelo")
    ap.add_argument("--suffix", default="_int8",
                    help="sufijo de salida (default: _int8, el que busca mission_pipeline.py)")
    args = ap.parse_args(argv)

    targets = [args.only] if args.only else MODELS
    for m in targets:
        src = Path(m)
        if not src.exists():
            print(f"[skip] {m} (no existe)")
            continue
        dst = src.with_name(src.stem + args.suffix + ".onnx")
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        mb_in, mb_out = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
        print(f"[OK] {dst.name}  {mb_in:.1f} MB → {mb_out:.1f} MB "
              f"(x{mb_in / max(mb_out, 1e-9):.1f} más liviano)")
        print("     ⚠ Peso ≠ velocidad. Medí el tiempo real en la Pi antes de")
        print("       afirmar una aceleración, y el acuerdo con:")
        print(f"         python evaluate.py --onnx {src} --compare {dst}")

    print("\n[LISTO] Modelos INT8 generados. Falta VALIDARLOS antes de volar:")
    print("  python validate_int8_mission.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
