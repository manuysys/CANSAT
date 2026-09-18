"""
CanSat La Base — Detección de Daños v2 (escala de 4 niveles tipo xBD/xView2).
Simula destrucción con 3 intensidades y detecta el nivel midiendo el colapso
de la confianza de la clase 'building' entre el ANTES y el DESPUÉS.

Uso:
    python simulate_disaster.py --image dataset/loveda_raw/Test/Urban/images_png/5921.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ONNX_PATH = "outputs/cansat_seg_deeplabv3plus_mobilenetv2.onnx"
IMG_SIZE = 320

# Escala xBD: (nivel, nombre, color BGR)
LEVELS = [
    (0, "SIN DAÑO",   (0, 255, 0)),
    (1, "DAÑO MENOR", (0, 255, 255)),
    (2, "DAÑO MAYOR", (0, 128, 255)),
    (3, "DESTRUIDO",  (0, 0, 255)),
]


def preprocess(img_bgr):
    small = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return norm.transpose(2, 0, 1)[np.newaxis, ...]


def building_prob(logits):
    """Probabilidad (softmax) de la clase building (1)."""
    m = logits.max(axis=0, keepdims=True)
    e = np.exp(logits - m)
    return e[1] / e.sum(axis=0)


def simulate_levels(img, mask_build, rng):
    """Corrompe cada edificio con una intensidad al azar (nivel real 1-3)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_build.astype(np.uint8), 8)
    img_after = img.copy().astype(np.float32)
    true_levels = np.zeros(img.shape[:2], dtype=np.uint8)
    noise = rng.normal(0, 1, img.shape).astype(np.float32)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        comp = (labels == i)
        lvl = int(rng.integers(1, 4))
        true_levels[comp] = lvl
        if lvl == 1:
            img_after[comp] = img_after[comp] * 0.75 + noise[comp] * 8
        elif lvl == 2:
            img_after[comp] = (img_after[comp] * 0.45
                               + np.array([50, 55, 60], np.float32)
                               + noise[comp] * 18)
        else:
            img_after[comp] = np.array([55, 65, 75], np.float32) + noise[comp] * 30
            sh = comp & (rng.random(img.shape[:2]) > 0.8)
            img_after[sh] *= 0.2

    return np.clip(img_after, 0, 255).astype(np.uint8), true_levels


def main():
    ap = argparse.ArgumentParser(description="Detección de daños v2 (niveles xBD)")
    ap.add_argument("--image",
                    default="dataset/loveda_raw/Test/Urban/images_png/5921.png")
    ap.add_argument("--out-dir", default="outputs/disaster_sim")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] No se pudo cargar {args.image}")
        raise SystemExit(1)
    h, w = img.shape[:2]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 62)
    print("  CanSat La Base — Detección de daños (escala xBD, 4 niveles)")
    print("=" * 62)

    sess = ort.InferenceSession(ONNX_PATH)

    logits_b = sess.run(["logits"], {"input": preprocess(img)})[0][0]
    p_before = cv2.resize(building_prob(logits_b), (w, h))
    seg_b = np.argmax(logits_b, axis=0)
    seg_b = cv2.resize(seg_b, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_build = (seg_b == 1)

    img_after, true_levels = simulate_levels(img, mask_build, rng)
    cv2.imwrite(str(out / "01_antes.jpg"), img)
    cv2.imwrite(str(out / "02_despues_simulado.jpg"), img_after)

    logits_a = sess.run(["logits"], {"input": preprocess(img_after)})[0][0]
    p_after = cv2.resize(building_prob(logits_a), (w, h))

    score = p_before - p_after
    det = np.zeros((h, w), dtype=np.uint8)
    det[mask_build & (score >= 0.15)] = 1
    det[mask_build & (score >= 0.40)] = 2
    det[mask_build & (score >= 0.70)] = 3

    print(f"\n  {'nivel':<12} {'real px':>9} {'detectado px':>13} {'acierto':>8}")
    for lvl, name, _ in LEVELS:
        real = int((true_levels == lvl).sum())
        dete = int((det == lvl).sum())
        ok = int(((true_levels == lvl) & (det == lvl)).sum())
        acc = ok / real * 100 if real else 0.0
        print(f"  {name:<12} {real:>9} {dete:>13} {acc:7.1f}%")

    bpx = max(1, int(mask_build.sum()))
    idx = (int((det == 1).sum()) + 2 * int((det == 2).sum())
           + 3 * int((det == 3).sum())) / (3 * bpx) * 100
    print(f"\n  ÍNDICE DE DAÑO GLOBAL : {idx:.1f}%")
    print(f"  RIESGO HUMANO (proxy) : "
          f"{(int((det == 2).sum()) + int((det == 3).sum())) / bpx * 100:.1f}% "
          f"de superficie construida severamente afectada")

    overlay = img_after.copy()
    for lvl, name, color in LEVELS:
        if lvl == 0:
            continue
        overlay[det == lvl] = (overlay[det == lvl] * 0.4
                               + np.array(color, np.float32) * 0.6).astype(np.uint8)
    cv2.rectangle(overlay, (0, 0), (w, 30 * len(LEVELS) + 10), (0, 0, 0), -1)
    for i, (lvl, name, color) in enumerate(LEVELS):
        cv2.rectangle(overlay, (10, 8 + i * 30), (26, 24 + i * 30), color, -1)
        cv2.putText(overlay, f"{name}: {int((det == lvl).sum())} px",
                    (34, 24 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
    cv2.imwrite(str(out / "04_mapa_niveles_xbd.jpg"), overlay)
    print(f"\n  Guardado: {out / '04_mapa_niveles_xbd.jpg'}")


if __name__ == "__main__":
    main()
