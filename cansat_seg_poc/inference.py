"""
CanSat La Base — Inferencia visual del modelo ONNX en imágenes reales.
v2: excluye píxeles sin datos (bordes negros de LoveDA) de los porcentajes.

Uso:
    python inference.py --image ruta/a/imagen.png
    python inference.py --test --num-samples 5
    python inference.py --test --splits Rural --id-start 4191 --id-end 4200
    python inference.py --folder dataset/loveda_raw/Test/Urban/images_png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import cv2

# onnxruntime se importa DENTRO de main(): mission_pipeline.py hace
# ``from inference import create_colored_mask`` y en la Pi Zero v1 no hay
# wheels de onnxruntime (usa el backend cv2.dnn de cansat.onnxio). Con el
# import en el tope, el script de vuelo no arrancaba en esa placa.

NUM_CLASSES = 5
CLASS_NAMES = ['vegetation', 'building', 'water', 'bare_ground', 'other']
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLOR_PALETTE = {
    0: (0, 200, 0),    # vegetation
    1: (180, 100, 0),  # building
    2: (255, 100, 0),  # water
    3: (0, 200, 255),  # bare_ground
    4: (128, 128, 128) # other
}

TEST_DIRS = {
    "Rural": Path("dataset/loveda_raw/Test/Rural/images_png"),
    "Urban": Path("dataset/loveda_raw/Test/Urban/images_png"),
}


def preprocess_image(image_path, img_size):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {image_path}")
    img_original = img.copy()
    resized = cv2.resize(img, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return norm.transpose(2, 0, 1)[np.newaxis, ...], img_original


def create_colored_mask(seg_map):
    mask = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for cid, color in COLOR_PALETTE.items():
        mask[seg_map == cid] = color
    return mask


def valid_mask_from(img_original, img_size, thresh):
    """1 = píxel con datos, 0 = casi negro (sin datos)."""
    if thresh <= 0:
        return np.ones((img_size, img_size), dtype=bool)
    valid = (img_original.max(axis=2) > thresh).astype(np.uint8)
    small = cv2.resize(valid, (img_size, img_size),
                       interpolation=cv2.INTER_NEAREST)
    return small.astype(bool)


def class_percentages(seg_map, valid):
    total = int(valid.sum())
    nodata = 100.0 * (1.0 - total / seg_map.size)
    if total == 0:
        return dict.fromkeys(range(NUM_CLASSES), 0.0), nodata
    pcts = {c: float(((seg_map == c) & valid).sum()) / total * 100.0
            for c in range(NUM_CLASSES)}
    return pcts, nodata


def process_one(sess, image_path, img_size, out_dir, alpha, nodata_thresh):
    tensor, img_original = preprocess_image(image_path, img_size)
    logits = sess.run(["logits"], {"input": tensor})[0]
    seg_map = np.argmax(logits[0], axis=0)

    valid = valid_mask_from(img_original, img_size, nodata_thresh)

    mask = create_colored_mask(seg_map)
    h_orig, w_orig = img_original.shape[:2]
    mask_resized = cv2.resize(mask, (w_orig, h_orig),
                              interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(img_original, 1 - alpha, mask_resized, alpha, 0)

    y = 20
    for cid, color in COLOR_PALETTE.items():
        cv2.rectangle(overlay, (10, y), (30, y + 20), color, -1)
        cv2.putText(overlay, CLASS_NAMES[cid], (40, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

    out_dir.mkdir(parents=True, exist_ok=True)
    base = image_path.stem
    cv2.imwrite(str(out_dir / f"{base}_segmented.jpg"), overlay)
    cv2.imwrite(str(out_dir / f"{base}_mask.jpg"), mask_resized)

    pcts, nodata = class_percentages(seg_map, valid)
    extra = f"  (sin datos: {nodata:.1f}%)" if nodata > 0.5 else ""
    print(f"\n  → [{image_path.parent.name}] {image_path.name}{extra}")
    for c in range(NUM_CLASSES):
        print(f"      {CLASS_NAMES[c]:<14} {pcts[c]:5.1f}%  {'█' * int(pcts[c] // 5)}")
    return pcts


def build_task_list(args):
    tasks = []
    if args.image:
        p = Path(args.image)
        if not p.is_file():
            print(f"[ERROR] No existe la imagen: {p}")
            sys.exit(1)
        tasks.append(("single", p, Path("outputs")))
    elif args.folder:
        folder = Path(args.folder)
        imgs = sorted(folder.glob("*.png"))
        if not imgs:
            print(f"[ERROR] No hay .png en {folder}")
            sys.exit(1)
        tasks += [("folder", p, Path("outputs/folder") / folder.name) for p in imgs]
    elif args.test:
        rng = random.Random(args.seed)
        for split in [s.strip() for s in args.splits.split(",")]:
            d = TEST_DIRS.get(split)
            if d is None or not d.is_dir():
                print(f"[ERROR] Split/carpeta faltante: {split}")
                sys.exit(1)
            imgs = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
            if args.id_start is not None or args.id_end is not None:
                lo = args.id_start if args.id_start is not None else 0
                hi = args.id_end if args.id_end is not None else 10**9
                imgs = [p for p in imgs if lo <= int(p.stem) <= hi]
            if not imgs:
                print(f"[WARN] Sin imágenes en {split} con ese filtro.")
                continue
            k = min(args.num_samples, len(imgs))
            chosen = sorted(rng.sample(imgs, k), key=lambda p: int(p.stem))
            print(f"  [{split}] {len(imgs)} disponibles, muestra: {k} "
                  f"→ IDs {[int(p.stem) for p in chosen]}")
            tasks += [(split, p, Path("outputs/test_inference") / split) for p in chosen]
    else:
        print("[ERROR] Especificá --image, --folder o --test")
        sys.exit(1)
    if not tasks:
        print("[ERROR] No quedó ninguna imagen para procesar.")
        sys.exit(1)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Inferencia visual CanSat (ONNX)")
    parser.add_argument("--image", default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--splits", default="Rural,Urban")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--id-start", type=int, default=None)
    parser.add_argument("--id-end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--onnx",
                        default="outputs/cansat_seg_terrain_v2.onnx")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--nodata-thresh", type=int, default=15,
                        help="Píxel con max canal <= este valor = sin datos (0 = desactivar)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CanSat La Base — Inferencia visual (ONNX) v2")
    print(f"  Modelo: {args.onnx}")

    try:
        import onnxruntime as ort
    except ImportError:
        sys.stderr.write(
            "[ERROR] Falta onnxruntime (pip install onnxruntime).\n"
            "  Este CLI evalúa con ORT. En placas sin wheels de ORT (Pi Zero v1)\n"
            "  usá mission_pipeline.py, que cae a cv2.dnn vía cansat.onnxio.\n"
        )
        return 1

    sess = ort.InferenceSession(args.onnx)
    tasks = build_task_list(args)
    print(f"  Imágenes a procesar: {len(tasks)}")
    print("=" * 60)

    results = []
    for split, path, out_dir in tasks:
        pcts = process_one(sess, path, args.img_size, out_dir,
                           args.alpha, args.nodata_thresh)
        results.append((split, path, pcts))

    print("\n" + "=" * 60)
    print("  RESUMEN GLOBAL (solo píxeles con datos)")
    print("=" * 60)
    accum = np.zeros(NUM_CLASSES)
    for _, _, pcts in results:
        for c in range(NUM_CLASSES):
            accum[c] += pcts[c]
    accum /= len(results)
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<14} {accum[c]:5.1f}%  {'█' * int(accum[c] // 5)}")

    print("\n  Dominante por imagen:")
    for split, path, pcts in results:
        dom = max(pcts, key=pcts.get)
        print(f"    {split:<6} {path.name:<12} → {CLASS_NAMES[dom]}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
