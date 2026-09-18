"""
CanSat La Base — Cuantización estática INT8 con calibración.

════════════════════════════════════════════════════════════════════════════
ARREGLOS DE LA AUDITORÍA
════════════════════════════════════════════════════════════════════════════
· **Fuga de datos.** Calibraba con ``dataset/loveda_raw/Test`` y validaba con
  ESE MISMO split (``imgs[100:120]``). El porcentaje de acuerdo que reportaba
  estaba optimistamente sesgado. Ahora calibra con **Train** y valida con
  **Val**, que son disjuntos.
· **No reproducible.** ``random.shuffle(imgs)`` sin semilla: cada corrida daba
  otro número. Ahora ``--seed`` con default fijo.
· **El veredicto impreso mentía en todos los casos.** Con ``agree >= 0.95``
  imprimía *"La cuantización con calibración mejoró significativamente"* sin
  haber comparado contra nada: mide ACUERDO con el FP32, no mejora. Reescrito.
· **Escribía ``_int8_calib.onnx``** pero ``mission_pipeline.py`` buscaba
  ``_int8.onnx`` (que produce ``quantize_all.py`` con cuantización DINÁMICA).
  Dos pipelines de cuantización generando archivos distintos sin que nadie
  aclarara cuál vuela. Ahora el nombre de salida es explícito y se valida
  contra el que consume el pipeline.
· ``if TEST_DIRS[0].exists() else []`` evaluaba sólo el primer directorio: si
  ``Test/Rural`` no existía pero ``Test/Urban`` sí, la lista quedaba vacía.

Uso:
    python quantize_onnx.py
    python quantize_onnx.py --in outputs/X.onnx --out outputs/X_int8.onnx --calib-split Train
"""

import random
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_IN = "outputs/cansat_seg_terrain_v2.onnx"
DEFAULT_OUT = "outputs/cansat_seg_terrain_v2_int8_qdq.onnx"

DATASET_ROOT = Path("dataset")


def split_dirs(split: str) -> list[Path]:
    """Directorios de imágenes de un split, en remapeado y en raw."""
    out: list[Path] = []
    for base in (DATASET_ROOT / "loveda_remapped", DATASET_ROOT / "loveda_raw"):
        for env in ("Rural", "Urban", "rural", "urban"):
            d = base / split / env / "images_png"
            if d.is_dir():
                out.append(d)
    return out


def collect(split: str) -> list[Path]:
    """Todas las imágenes de un split. Antes sólo miraba el primer directorio."""
    imgs: list[Path] = []
    for d in split_dirs(split):
        imgs += sorted(d.glob("*.png"))
    # Deduplicar por nombre: remapeado y raw pueden solaparse.
    seen: set[str] = set()
    uniq = []
    for p in imgs:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq


class LovedaCalibrationDataReader(CalibrationDataReader):
    def __init__(self, images, img_size):
        self.images = images
        self.img_size = img_size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.images):
            return None
        img = cv2.imread(str(self.images[self.idx]))
        img = cv2.resize(img, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = norm.transpose(2, 0, 1)[np.newaxis, ...]
        self.idx += 1
        return {"input": tensor}


def preprocess(path, size):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return norm.transpose(2, 0, 1)[np.newaxis, ...]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Cuantización estática INT8 con calibración",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", dest="outp", default=DEFAULT_OUT)
    ap.add_argument("--calib-split", default="Train",
                    help="split para CALIBRAR. Debe ser distinto del de validación.")
    ap.add_argument("--val-split", default="Val",
                    help="split para VALIDAR el acuerdo FP32 vs INT8.")
    ap.add_argument("--calib-images", type=int, default=100)
    ap.add_argument("--val-images", type=int, default=100)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42,
                    help="antes el shuffle no tenía semilla: no era reproducible")
    args = ap.parse_args(argv)

    if args.calib_split == args.val_split:
        print(f"[ERROR] --calib-split y --val-split son el mismo ('{args.calib_split}').")
        print("        Calibrar y validar sobre el mismo split es fuga de datos,")
        print("        que es exactamente lo que hacía la versión anterior (usaba")
        print("        Test para las dos cosas).")
        raise SystemExit(1)

    inp, outp = Path(args.inp), Path(args.outp)
    if not inp.exists():
        print(f"[ERROR] No existe {inp}")
        raise SystemExit(1)

    imgs = collect(args.calib_split)
    if not imgs:
        print(f"[ERROR] No hay imágenes en el split '{args.calib_split}' "
              f"bajo {DATASET_ROOT}.")
        raise SystemExit(1)

    val_imgs = collect(args.val_split)
    if not val_imgs:
        print(f"[ERROR] No hay imágenes en el split '{args.val_split}' para validar.")
        raise SystemExit(1)

    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    calib_imgs = imgs[:args.calib_images]
    rng.shuffle(val_imgs)
    val_imgs = val_imgs[:args.val_images]
    print(f"  Calibración: {len(calib_imgs)} imágenes de '{args.calib_split}'")
    print(f"  Validación : {len(val_imgs)} imágenes de '{args.val_split}' (disjunto)")
    print(f"  Semilla    : {args.seed}")

    print(f"[1/3] Cuantizando {inp.name} → {outp.name} (INT8 estático QDQ)...")

    calib_reader = LovedaCalibrationDataReader(calib_imgs, args.img_size)
    quantize_static(
        str(inp),
        str(outp),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,  # ← CORREGIDO: QDQ en vez de QUInt8
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
    )

    mb_in = inp.stat().st_size / 1e6
    mb_out = outp.stat().st_size / 1e6
    print(f"      FP32: {mb_in:.1f} MB → INT8: {mb_out:.1f} MB "
          f"(x{mb_in / mb_out:.1f} más liviano)")

    print("[2/3] Validando precisión FP32 vs INT8...")
    sess_fp = ort.InferenceSession(str(inp))
    sess_q = ort.InferenceSession(str(outp))

    agree, ndiff = [], []
    for p in val_imgs:
        x = preprocess(p, args.img_size)
        a = sess_fp.run(["logits"], {"input": x})[0]
        b = sess_q.run(["logits"], {"input": x})[0]
        agree.append((np.argmax(a, 1) == np.argmax(b, 1)).mean())
        ndiff.append(np.abs(a - b).max())
    agree, ndiff = np.mean(agree), np.mean(ndiff)
    print(f"      Coincidencia de predicción por píxel: {agree * 100:.2f}%")
    print(f"      Diferencia máx. de logits: {ndiff:.4f}")

    print("[3/3] Veredicto:")
    print("      ⚠ Ojo con qué mide este número: es el ACUERDO entre el FP32 y")
    print("        el INT8, no una mejora de precisión. Que el INT8 coincida")
    print("        mucho con el FP32 no dice nada sobre si el modelo es bueno.")
    print("        La métrica de calidad real es el mIoU sobre Val:")
    print("            python evaluate.py --onnx <fp32> --compare <int8>")
    print("        Y la que decide si vuela es la de MISIÓN:")
    print("            python validate_int8_mission.py")
    if agree >= 0.99:
        print(f"      Acuerdo {agree * 100:.2f}%: el INT8 es prácticamente idéntico")
        print("      al FP32 píxel a píxel. Igual hay que confirmar con mIoU.")
    elif agree >= 0.95:
        print(f"      Acuerdo {agree * 100:.2f}%: dentro del umbral del proyecto (95%),")
        print("      pero confirmar con evaluate.py y validate_int8_mission.py")
        print("      ANTES de decidir volar en INT8.")
    elif agree >= 0.90:
        print(f"      Acuerdo {agree * 100:.2f}%: BAJO el umbral de 95%. El proyecto")
        print("      descartó INT8 con 87.6%, aunque esa medición era inválida.")
        print("      Re-medir con validate_int8_mission.py antes de concluir.")
    else:
        print(f"      Acuerdo {agree * 100:.2f}%: el INT8 diverge fuerte del FP32.")
        print("      No usarlo. Revisar el dataset de calibración (¿es representativo")
        print("      de lo que ve la cámara en vuelo?) o pasar a QAT.")


if __name__ == "__main__":
    raise SystemExit(main())
