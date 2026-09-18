"""
CanSat La Base — Test-Time Augmentation para segmentación.
Aplica 4 augmentaciones (original + 3 flips) y promedia predicciones.
Sube mIoU ~2-5% sin reentrenar.

Uso:
    python tta_inference.py --image dataset/loveda_raw/Test/Urban/images_png/5861.png
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prep(bgr, size=320):
    small = cv2.resize(bgr, (size, size))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis, ...]


def tta_segment(sess, bgr, size=320):
    """Segmentación con TTA: 4 augmentaciones + promedio."""
    augs = [
        (bgr, None),
        (cv2.flip(bgr, 1), 'h'),
        (cv2.flip(bgr, 0), 'v'),
        (cv2.flip(bgr, -1), 'hv'),
    ]
    logits_sum = None
    for img, flip in augs:
        x = prep(img, size)
        logits = sess.run(["logits"], {"input": x})[0]
        if flip == 'h':
            logits = logits[:, :, :, ::-1]
        elif flip == 'v':
            logits = logits[:, :, ::-1, :]
        elif flip == 'hv':
            logits = logits[:, :, ::-1, ::-1]
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum += logits
    return logits_sum / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--onnx", default="outputs/cansat_seg_terrain_v2.onnx")
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx)
    bgr = cv2.imread(args.image)

    # Normal (sin TTA)
    x = prep(bgr, args.size)
    logits_normal = sess.run(["logits"], {"input": x})[0]
    seg_normal = np.argmax(logits_normal[0], axis=0)

    # Con TTA
    logits_tta = tta_segment(sess, bgr, args.size)
    seg_tta = np.argmax(logits_tta[0], axis=0)

    classes = ['vegetation', 'building', 'water', 'bare_ground', 'other']
    print("Comparación Normal vs TTA:")
    print("-" * 50)
    for i, c in enumerate(classes):
        p_normal = float((seg_normal == i).mean()) * 100
        p_tta = float((seg_tta == i).mean()) * 100
        diff = p_tta - p_normal
        print(f"  {c:15s}: normal {p_normal:5.1f}% | TTA {p_tta:5.1f}% ({diff:+.1f})")

    # Guardar comparación visual
    vis = np.hstack([seg_normal, seg_tta]) * 50
    cv2.imwrite("outputs/tta_compare.jpg", vis.astype(np.uint8))
    print("\n[OK] Comparación visual: outputs/tta_compare.jpg")


if __name__ == "__main__":
    main()
